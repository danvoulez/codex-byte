use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{Extension, Path, State},
    http::{HeaderMap, HeaderValue, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const DEFAULT_APP: &str = "default";
const DEFAULT_TENANT: &str = "default";
const RATE_LIMIT: usize = 20;
const RATE_WINDOW_SECS: u64 = 60;
const IDEM_TTL_SECS: u64 = 600;
const IDEM_MAX_ENTRIES: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Scope {
    pub app: String,
    pub tenant: String,
}

impl Scope {
    pub fn new(app: impl Into<String>, tenant: impl Into<String>) -> Self {
        Self {
            app: app.into(),
            tenant: tenant.into(),
        }
    }

    pub fn key_prefix(&self) -> String {
        format!("a/{}/t/{}/", self.app, self.tenant)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Wa,
    Transition,
    Wf,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Decision {
    Pass,
    Retry,
    Deny,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReceiptV1 {
    pub scope: Scope,
    pub phase: Phase,
    pub body_cid: String,
    pub runtime_did: String,
    pub runtime_sha256: Option<String>,
    pub jws_alg: String,
    pub jws_kid: String,
    pub jws_payload_hex: String,
    pub parent_cid: Option<String>,
    pub decision: Option<Decision>,
    pub ts_ms: i64,
}

#[derive(Clone, Debug, Serialize)]
struct ReceiptEvent {
    key: String,
    phase: Phase,
    ts_ms: i64,
}

#[derive(Debug, Serialize)]
pub struct AppErrorEnvelope {
    error: AppError,
}

#[derive(Debug, Serialize)]
pub struct AppError {
    code: String,
    message: String,
    request_id: String,
    retry_after: u64,
}

impl AppError {
    fn new(code: &str, message: &str, request_id: String, retry_after: u64) -> Self {
        Self {
            code: code.to_string(),
            message: message.to_string(),
            request_id,
            retry_after,
        }
    }
}

impl IntoResponse for AppErrorEnvelope {
    fn into_response(self) -> Response {
        let status = match self.error.code.as_str() {
            "UNAUTHORIZED" => StatusCode::UNAUTHORIZED,
            "FORBIDDEN_SCOPE" | "CORS_DENIED" => StatusCode::FORBIDDEN,
            "NOT_FOUND" => StatusCode::NOT_FOUND,
            "RATE_LIMITED" => StatusCode::TOO_MANY_REQUESTS,
            _ => StatusCode::BAD_REQUEST,
        };

        let mut resp = (status, Json(self)).into_response();
        let request_id = resp
            .extensions()
            .get::<String>()
            .cloned()
            .unwrap_or_default();
        if !request_id.is_empty() {
            resp.headers_mut()
                .insert("x-request-id", HeaderValue::from_str(&request_id).unwrap());
        }
        resp
    }
}

#[derive(Clone)]
pub struct AppState {
    ledger: Arc<Mutex<HashMap<String, ReceiptV1>>>,
    events: Arc<Mutex<Vec<ReceiptEvent>>>,
    cors: Arc<CorsConfig>,
    idempotency: Arc<Mutex<IdempotencyStore>>,
    rate: Arc<Mutex<HashMap<String, VecDeque<Instant>>>>,
    policies: Arc<Vec<Box<dyn Policy + Send + Sync>>>,
}

#[derive(Clone)]
struct RequestMeta {
    request_id: String,
    origin: Option<String>,
    rate: Option<RateLimitState>,
}

#[derive(Clone)]
struct RateLimitState {
    limit: usize,
    remaining: usize,
}

#[derive(Clone)]
pub struct CorsConfig {
    pub global_allowed: Vec<String>,
    pub app_allowed: HashMap<String, Vec<String>>,
    pub app_tenant_allowed: HashMap<(String, String), Vec<String>>,
}

impl CorsConfig {
    pub fn is_origin_allowed(&self, origin: &str, scope: Option<&Scope>) -> bool {
        if let Some(s) = scope {
            if let Some(list) = self
                .app_tenant_allowed
                .get(&(s.app.clone(), s.tenant.clone()))
            {
                return list.iter().any(|o| o == origin);
            }
            if let Some(list) = self.app_allowed.get(&s.app) {
                return list.iter().any(|o| o == origin);
            }
        }
        self.global_allowed.iter().any(|o| o == origin)
    }
}

impl Default for AppState {
    fn default() -> Self {
        let mut app_allowed = HashMap::new();
        app_allowed.insert("acme".into(), vec!["https://acme.app".into()]);
        let mut app_tenant_allowed = HashMap::new();
        app_tenant_allowed.insert(
            ("acme".into(), "pro".into()),
            vec!["https://tenant.acme.app".into()],
        );

        Self {
            ledger: Arc::new(Mutex::new(HashMap::new())),
            events: Arc::new(Mutex::new(vec![])),
            cors: Arc::new(CorsConfig {
                global_allowed: vec!["https://global.app".into()],
                app_allowed,
                app_tenant_allowed,
            }),
            idempotency: Arc::new(Mutex::new(IdempotencyStore::new(
                IDEM_MAX_ENTRIES,
                Duration::from_secs(IDEM_TTL_SECS),
            ))),
            rate: Arc::new(Mutex::new(HashMap::new())),
            policies: Arc::new(vec![
                Box::new(DenyPolicy),
                Box::new(RetryPolicy),
                Box::new(AllowAllPolicy),
            ]),
        }
    }
}

trait Policy {
    fn id(&self) -> &'static str;
    fn decide(&self, ctx: &PolicyCtx<'_>) -> Decision;
}

struct PolicyCtx<'a> {
    scope: &'a Scope,
    manifest: &'a Value,
}

struct AllowAllPolicy;
struct RetryPolicy;
struct DenyPolicy;

impl Policy for AllowAllPolicy {
    fn id(&self) -> &'static str {
        "allow-all:v1"
    }

    fn decide(&self, _ctx: &PolicyCtx<'_>) -> Decision {
        Decision::Pass
    }
}

impl Policy for RetryPolicy {
    fn id(&self) -> &'static str {
        "retry-on-manifest:v1"
    }

    fn decide(&self, ctx: &PolicyCtx<'_>) -> Decision {
        if ctx
            .manifest
            .get("retry")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            Decision::Retry
        } else {
            Decision::Pass
        }
    }
}

impl Policy for DenyPolicy {
    fn id(&self) -> &'static str {
        "deny-on-manifest:v1"
    }

    fn decide(&self, ctx: &PolicyCtx<'_>) -> Decision {
        if ctx
            .manifest
            .get("deny")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            Decision::Deny
        } else if ctx.scope.app == "blocked" {
            Decision::Deny
        } else {
            Decision::Pass
        }
    }
}

#[derive(Deserialize)]
pub struct ExecuteRequest {
    manifest: Value,
    vars: Value,
    options: Option<ExecuteOptions>,
}

#[derive(Deserialize)]
pub struct ExecuteOptions {
    ghost: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ExecuteResponse {
    decision: Decision,
    tip_cid: String,
    links: HashMap<String, String>,
    runtime: RuntimeInfo,
    policy_trace: Vec<PolicyTrace>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RuntimeInfo {
    did: String,
    sha256: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PolicyTrace {
    id: String,
    decision: Decision,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/a/:app/t/:tenant/v1/execute", post(execute_scoped))
        .route("/a/:app/t/:tenant/v1/receipt/:cid", get(get_receipt_scoped))
        .route("/a/:app/t/:tenant/v1/receipts", get(list_receipts_scoped))
        .route("/a/:app/t/:tenant/v1/audit", get(audit_scoped))
        .route(
            "/a/:app/t/:tenant/v1/transition/:cid",
            get(get_transition_scoped),
        )
        .route("/v1/execute", post(execute_legacy))
        .route("/v1/receipt/:cid", get(get_receipt_legacy))
        .route("/v1/receipts", get(list_receipts_legacy))
        .route("/v1/audit", get(audit_legacy))
        .route("/v1/transition/:cid", get(get_transition_legacy))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            perimeter_middleware,
        ))
        .with_state(state)
}

async fn perimeter_middleware(
    State(state): State<AppState>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    let request_id = mk_request_id(req.method(), req.uri().path());
    let scope = parse_scope_from_path(req.uri().path());
    let origin = req
        .headers()
        .get("origin")
        .and_then(|v| v.to_str().ok())
        .map(ToOwned::to_owned);

    if let Some(origin) = origin.as_deref() {
        if !state.cors.is_origin_allowed(origin, scope.as_ref()) {
            return app_error(
                "CORS_DENIED",
                "origin is not allowed",
                &request_id,
                0,
                Some(origin),
            )
            .into_response();
        }

        if req.method() == Method::OPTIONS {
            let mut r = Response::new(Body::empty());
            *r.status_mut() = StatusCode::NO_CONTENT;
            add_cors_headers(r.headers_mut(), origin);
            r.headers_mut()
                .insert("x-request-id", HeaderValue::from_str(&request_id).unwrap());
            return r;
        }
    }

    if req.method() != Method::OPTIONS {
        if let Err(err) = check_auth(req.headers(), scope.as_ref(), request_id.clone()) {
            return with_cors(err.into_response(), origin.as_deref());
        }
    }

    if let Some(sc) = scope {
        let client_id = extract_client(req.headers()).unwrap_or_else(|| "anonymous".to_string());
        let rate_key = format!("rl:{}:{}:{}", sc.app, sc.tenant, client_id);
        let rate_state = apply_rate_limit(&state, rate_key);

        if let Some(retry_after) = rate_state.retry_after {
            let mut err = app_error(
                "RATE_LIMITED",
                "too many requests",
                &request_id,
                retry_after,
                origin.as_deref(),
            )
            .into_response();
            err.headers_mut().insert(
                "retry-after",
                HeaderValue::from_str(&retry_after.to_string()).unwrap(),
            );
            return err;
        }

        req.extensions_mut().insert(sc);
        req.extensions_mut().insert(RequestMeta {
            request_id,
            origin,
            rate: Some(RateLimitState {
                limit: rate_state.limit,
                remaining: rate_state.remaining,
            }),
        });
    } else {
        req.extensions_mut().insert(RequestMeta {
            request_id,
            origin,
            rate: None,
        });
    }

    let meta = req.extensions().get::<RequestMeta>().cloned();
    let mut response = next.run(req).await;
    if let Some(meta) = meta {
        response.headers_mut().insert(
            "x-request-id",
            HeaderValue::from_str(&meta.request_id).unwrap(),
        );
        if let Some(rate) = meta.rate {
            response.headers_mut().insert(
                "x-ratelimit-limit",
                HeaderValue::from_str(&rate.limit.to_string()).unwrap(),
            );
            response.headers_mut().insert(
                "x-ratelimit-remaining",
                HeaderValue::from_str(&rate.remaining.to_string()).unwrap(),
            );
        }
        if let Some(origin) = meta.origin {
            add_cors_headers(response.headers_mut(), &origin);
        }
    }
    response
}

fn mk_request_id(method: &Method, path: &str) -> String {
    let seed = format!(
        "{}:{}:{}",
        method,
        path,
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    );
    format!("req_{}", blake3::hash(seed.as_bytes()).to_hex())
}

fn with_cors(mut response: Response, origin: Option<&str>) -> Response {
    if let Some(origin) = origin {
        add_cors_headers(response.headers_mut(), origin);
    }
    response
}

fn add_cors_headers(headers: &mut HeaderMap, origin: &str) {
    headers.insert(
        "access-control-allow-origin",
        HeaderValue::from_str(origin).unwrap(),
    );
    headers.insert(
        "access-control-allow-methods",
        HeaderValue::from_static("GET,POST,OPTIONS"),
    );
    headers.insert(
        "access-control-allow-headers",
        HeaderValue::from_static("content-type,authorization,idempotency-key"),
    );
    headers.insert(
        "access-control-expose-headers",
        HeaderValue::from_static(
            "x-ratelimit-limit,x-ratelimit-remaining,retry-after,x-request-id",
        ),
    );
    headers.insert("vary", HeaderValue::from_static("origin"));
}

fn parse_scope_from_path(path: &str) -> Option<Scope> {
    let parts: Vec<_> = path.split('/').collect();
    if parts.len() > 5 && parts[1] == "a" && parts[3] == "t" {
        return Some(Scope::new(parts[2], parts[4]));
    }
    if path.starts_with("/v1/") {
        return Some(Scope::new(DEFAULT_APP, DEFAULT_TENANT));
    }
    None
}

fn check_auth(
    headers: &HeaderMap,
    scope: Option<&Scope>,
    request_id: String,
) -> Result<(), AppErrorEnvelope> {
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    let token = auth.strip_prefix("Bearer ").unwrap_or_default();

    let Some(scope) = scope else {
        return Ok(());
    };

    if token.is_empty() {
        return Err(AppErrorEnvelope {
            error: AppError::new("UNAUTHORIZED", "missing bearer token", request_id, 0),
        });
    }

    let expected = format!("{}:{}", scope.app, scope.tenant);
    if token != "root" && token != expected {
        return Err(AppErrorEnvelope {
            error: AppError::new(
                "FORBIDDEN_SCOPE",
                &format!(
                    "Token not allowed for scope a:{}/t:{}",
                    scope.app, scope.tenant
                ),
                request_id,
                0,
            ),
        });
    }

    Ok(())
}

fn extract_client(headers: &HeaderMap) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(ToOwned::to_owned)
}

struct RateLimitResult {
    limit: usize,
    remaining: usize,
    retry_after: Option<u64>,
}

fn apply_rate_limit(state: &AppState, key: String) -> RateLimitResult {
    let mut map = state.rate.lock().unwrap();
    let now = Instant::now();
    let window = Duration::from_secs(RATE_WINDOW_SECS);
    let bucket = map.entry(key).or_default();

    while bucket
        .front()
        .is_some_and(|t| now.duration_since(*t) > window)
    {
        bucket.pop_front();
    }

    if bucket.len() >= RATE_LIMIT {
        return RateLimitResult {
            limit: RATE_LIMIT,
            remaining: 0,
            retry_after: Some(RATE_WINDOW_SECS),
        };
    }

    bucket.push_back(now);
    RateLimitResult {
        limit: RATE_LIMIT,
        remaining: RATE_LIMIT.saturating_sub(bucket.len()),
        retry_after: None,
    }
}

struct IdempotencyEntry {
    body_sha: String,
    response: Value,
    created_at: Instant,
}

struct IdempotencyStore {
    entries: HashMap<String, IdempotencyEntry>,
    order: VecDeque<String>,
    max_entries: usize,
    ttl: Duration,
}

impl IdempotencyStore {
    fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            max_entries,
            ttl,
        }
    }

    fn cleanup(&mut self) {
        let now = Instant::now();
        self.entries
            .retain(|_, v| now.duration_since(v.created_at) <= self.ttl);
        self.order.retain(|k| self.entries.contains_key(k));

        while self.entries.len() > self.max_entries {
            if let Some(old) = self.order.pop_front() {
                self.entries.remove(&old);
            } else {
                break;
            }
        }
    }

    fn get(&mut self, key: &str) -> Option<&IdempotencyEntry> {
        self.cleanup();
        self.entries.get(key)
    }

    fn insert(&mut self, key: String, body_sha: String, response: Value) {
        self.cleanup();
        self.order.push_back(key.clone());
        self.entries.insert(
            key,
            IdempotencyEntry {
                body_sha,
                response,
                created_at: Instant::now(),
            },
        );
        self.cleanup();
    }
}

async fn execute_scoped(
    Extension(meta): Extension<RequestMeta>,
    Path((app, tenant)): Path<(String, String)>,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<ExecuteRequest>,
) -> Result<Json<ExecuteResponse>, AppErrorEnvelope> {
    execute_inner(meta, Scope::new(app, tenant), state, headers, payload).await
}

async fn execute_legacy(
    Extension(meta): Extension<RequestMeta>,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<ExecuteRequest>,
) -> Result<Json<ExecuteResponse>, AppErrorEnvelope> {
    execute_inner(
        meta,
        Scope::new(DEFAULT_APP, DEFAULT_TENANT),
        state,
        headers,
        payload,
    )
    .await
}

async fn execute_inner(
    meta: RequestMeta,
    scope: Scope,
    state: AppState,
    headers: HeaderMap,
    payload: ExecuteRequest,
) -> Result<Json<ExecuteResponse>, AppErrorEnvelope> {
    let ghost = payload
        .options
        .as_ref()
        .and_then(|o| o.ghost)
        .unwrap_or(false);

    let canonical = canonical_input(&payload.manifest, &payload.vars, ghost);
    let body_sha = blake3::hash(canonical.as_bytes()).to_hex().to_string();

    if let Some(idem) = headers
        .get("idempotency-key")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
    {
        let scoped_path = format!("/a/{}/t/{}/v1/execute", scope.app, scope.tenant);
        let store_key = format!(
            "idem:{}:{}:POST:{}:{}",
            scope.app, scope.tenant, scoped_path, idem
        );

        let mut store = state.idempotency.lock().unwrap();
        if let Some(existing) = store.get(&store_key) {
            if existing.body_sha == body_sha {
                return Ok(Json(
                    serde_json::from_value(existing.response.clone()).unwrap(),
                ));
            }
            return Err(app_error(
                "IDEMPOTENCY_CONFLICT",
                "idempotency key already used with a different payload",
                &meta.request_id,
                0,
                meta.origin.as_deref(),
            ));
        }

        let (decision, policy_trace) = evaluate_policies(&state, &scope, &payload.manifest);
        let wf_cid = persist_receipts(&state, &scope, &canonical, decision.clone());

        let response = mk_execute_response(&scope, decision, wf_cid, policy_trace);
        store.insert(
            store_key,
            body_sha,
            serde_json::to_value(&response).expect("serialize execute response"),
        );
        return Ok(Json(response));
    }

    let (decision, policy_trace) = evaluate_policies(&state, &scope, &payload.manifest);
    let wf_cid = persist_receipts(&state, &scope, &canonical, decision.clone());
    Ok(Json(mk_execute_response(
        &scope,
        decision,
        wf_cid,
        policy_trace,
    )))
}

fn canonical_input(manifest: &Value, vars: &Value, ghost: bool) -> String {
    let mut root = serde_json::Map::new();
    root.insert("manifest".into(), canonicalize_value(manifest));
    root.insert("vars".into(), canonicalize_value(vars));
    root.insert("ghost".into(), Value::Bool(ghost));
    serde_json::to_string(&Value::Object(root)).expect("canonical json")
}

fn canonicalize_value(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<_> = map.keys().cloned().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for k in keys {
                out.insert(k.clone(), canonicalize_value(&map[&k]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonicalize_value).collect()),
        _ => v.clone(),
    }
}

fn evaluate_policies(
    state: &AppState,
    scope: &Scope,
    manifest: &Value,
) -> (Decision, Vec<PolicyTrace>) {
    let ctx = PolicyCtx { scope, manifest };
    let mut trace = vec![];
    for p in state.policies.iter() {
        let decision = p.decide(&ctx);
        trace.push(PolicyTrace {
            id: p.id().to_string(),
            decision: decision.clone(),
        });
        if decision != Decision::Pass {
            return (decision, trace);
        }
    }
    (Decision::Pass, trace)
}

fn persist_receipts(
    state: &AppState,
    scope: &Scope,
    canonical: &str,
    decision: Decision,
) -> String {
    let wa_cid = cid_from_bytes(format!("{}:wa", canonical).as_bytes());
    let tr_cid = cid_from_bytes(format!("{}:{}:tr", canonical, wa_cid).as_bytes());
    let wf_cid = cid_from_bytes(format!("{}:{}:{:?}:wf", canonical, tr_cid, decision).as_bytes());

    let receipts = vec![
        mk_receipt(scope.clone(), Phase::Wa, wa_cid.clone(), None, None),
        mk_receipt(
            scope.clone(),
            Phase::Transition,
            tr_cid.clone(),
            Some(wa_cid),
            None,
        ),
        mk_receipt(
            scope.clone(),
            Phase::Wf,
            wf_cid.clone(),
            Some(tr_cid),
            Some(decision),
        ),
    ];

    let mut ledger = state.ledger.lock().unwrap();
    let mut events = state.events.lock().unwrap();
    for r in receipts {
        let key = format!("{}{}", scope.key_prefix(), r.body_cid);
        ledger.insert(key.clone(), r.clone());
        events.push(ReceiptEvent {
            key,
            phase: r.phase,
            ts_ms: Utc::now().timestamp_millis(),
        });
    }

    wf_cid
}

fn mk_execute_response(
    scope: &Scope,
    decision: Decision,
    wf_cid: String,
    policy_trace: Vec<PolicyTrace>,
) -> ExecuteResponse {
    ExecuteResponse {
        decision,
        tip_cid: wf_cid.clone(),
        links: HashMap::from([
            (
                "self".to_string(),
                format!("/a/{}/t/{}/v1/receipt/{}", scope.app, scope.tenant, wf_cid),
            ),
            (
                "verify".to_string(),
                format!("/a/{}/t/{}/v1/verify/{}", scope.app, scope.tenant, wf_cid),
            ),
        ]),
        runtime: RuntimeInfo {
            did: "did:key:z6MkRuntime".to_string(),
            sha256: "runtime-sha256".to_string(),
        },
        policy_trace,
    }
}

fn mk_receipt(
    scope: Scope,
    phase: Phase,
    cid: String,
    parent: Option<String>,
    decision: Option<Decision>,
) -> ReceiptV1 {
    ReceiptV1 {
        scope,
        phase,
        body_cid: cid.clone(),
        runtime_did: "did:key:z6MkRuntime".into(),
        runtime_sha256: Some("runtime-sha256".into()),
        jws_alg: "Ed25519".into(),
        jws_kid: "did:key:z6MkRuntime".into(),
        jws_payload_hex: hex::encode(cid.as_bytes()),
        parent_cid: parent,
        decision,
        ts_ms: Utc::now().timestamp_millis(),
    }
}

fn cid_from_bytes(bytes: &[u8]) -> String {
    format!("b3:{}", blake3::hash(bytes).to_hex())
}

fn normalize_cid(cid: &str) -> String {
    cid.replace("%3A", ":").replace("%3a", ":")
}

async fn get_receipt_scoped(
    Extension(meta): Extension<RequestMeta>,
    Path((app, tenant, cid)): Path<(String, String, String)>,
    State(state): State<AppState>,
) -> Result<Json<ReceiptV1>, AppErrorEnvelope> {
    get_receipt_inner(meta, Scope::new(app, tenant), cid, state)
}

async fn get_receipt_legacy(
    Extension(meta): Extension<RequestMeta>,
    Path(cid): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<ReceiptV1>, AppErrorEnvelope> {
    get_receipt_inner(meta, Scope::new(DEFAULT_APP, DEFAULT_TENANT), cid, state)
}

fn get_receipt_inner(
    meta: RequestMeta,
    scope: Scope,
    cid: String,
    state: AppState,
) -> Result<Json<ReceiptV1>, AppErrorEnvelope> {
    let normalized = normalize_cid(&cid);
    let key = format!("{}{}", scope.key_prefix(), normalized);
    let ledger = state.ledger.lock().unwrap();
    let Some(r) = ledger.get(&key) else {
        return Err(app_error(
            "NOT_FOUND",
            "receipt not found",
            &meta.request_id,
            0,
            meta.origin.as_deref(),
        ));
    };
    Ok(Json(r.clone()))
}

async fn get_transition_scoped(
    Extension(meta): Extension<RequestMeta>,
    Path((app, tenant, cid)): Path<(String, String, String)>,
    State(state): State<AppState>,
) -> Result<Json<ReceiptV1>, AppErrorEnvelope> {
    get_transition_inner(meta, Scope::new(app, tenant), cid, state)
}

async fn get_transition_legacy(
    Extension(meta): Extension<RequestMeta>,
    Path(cid): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<ReceiptV1>, AppErrorEnvelope> {
    get_transition_inner(meta, Scope::new(DEFAULT_APP, DEFAULT_TENANT), cid, state)
}

fn get_transition_inner(
    meta: RequestMeta,
    scope: Scope,
    cid: String,
    state: AppState,
) -> Result<Json<ReceiptV1>, AppErrorEnvelope> {
    let receipt = get_receipt_inner(meta.clone(), scope, cid, state)?;
    if receipt.phase != Phase::Transition {
        return Err(app_error(
            "NOT_FOUND",
            "transition receipt not found",
            &meta.request_id,
            0,
            meta.origin.as_deref(),
        ));
    }
    Ok(receipt)
}

async fn list_receipts_scoped(
    Path((app, tenant)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Json<Value> {
    list_receipts_inner(Scope::new(app, tenant), state)
}

async fn list_receipts_legacy(State(state): State<AppState>) -> Json<Value> {
    list_receipts_inner(Scope::new(DEFAULT_APP, DEFAULT_TENANT), state)
}

fn list_receipts_inner(scope: Scope, state: AppState) -> Json<Value> {
    let prefix = scope.key_prefix();
    let keys: Vec<String> = state
        .ledger
        .lock()
        .unwrap()
        .keys()
        .filter(|k| k.starts_with(&prefix))
        .map(|k| k.strip_prefix(&prefix).unwrap_or(k.as_str()).to_string())
        .map(|cid| format!("{}:{}:{}", scope.app, scope.tenant, cid))
        .collect();
    Json(json!({ "keys": keys }))
}

async fn audit_scoped(
    Path((app, tenant)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Json<Value> {
    audit_inner(Scope::new(app, tenant), state)
}

async fn audit_legacy(State(state): State<AppState>) -> Json<Value> {
    audit_inner(Scope::new(DEFAULT_APP, DEFAULT_TENANT), state)
}

fn audit_inner(scope: Scope, state: AppState) -> Json<Value> {
    let prefix = scope.key_prefix();
    let events: Vec<Value> = state
        .events
        .lock()
        .unwrap()
        .iter()
        .filter(|e| e.key.starts_with(&prefix))
        .map(|e| {
            let cid = e
                .key
                .strip_prefix(&prefix)
                .unwrap_or(e.key.as_str())
                .to_string();
            json!({
                "key": format!("{}:{}:{}", scope.app, scope.tenant, cid),
                "phase": e.phase,
                "ts_ms": e.ts_ms,
            })
        })
        .collect();
    Json(json!({ "events": events }))
}

fn app_error(
    code: &str,
    message: &str,
    request_id: &str,
    retry_after: u64,
    _origin: Option<&str>,
) -> AppErrorEnvelope {
    AppErrorEnvelope {
        error: AppError::new(code, message, request_id.to_string(), retry_after),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn auth_for(app: &str, tenant: &str) -> String {
        format!("Bearer {}:{}", app, tenant)
    }

    async fn exec(
        app: &Router,
        app_name: &str,
        tenant: &str,
        body: &str,
        idem: Option<&str>,
    ) -> (StatusCode, Value, HeaderMap) {
        let mut b = Request::builder()
            .method("POST")
            .uri(format!("/a/{app_name}/t/{tenant}/v1/execute"))
            .header("authorization", auth_for(app_name, tenant))
            .header("content-type", "application/json");
        if let Some(id) = idem {
            b = b.header("idempotency-key", id);
        }
        let req = b.body(Body::from(body.to_string())).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let headers = resp.headers().clone();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        (status, json, headers)
    }

    #[tokio::test]
    async fn deterministic_execute_10x() {
        let app = build_router(AppState::default());
        let body = r#"{"manifest":{},"vars":{"doc":"x"},"options":{"ghost":true}}"#;

        let mut tip = String::new();
        for i in 0..10 {
            let (status, out, _) = exec(&app, "acme", "pro", body, None).await;
            assert_eq!(status, StatusCode::OK);
            if i == 0 {
                tip = out["tip_cid"].as_str().unwrap().to_string();
            } else {
                assert_eq!(tip, out["tip_cid"].as_str().unwrap());
            }
        }
    }

    #[tokio::test]
    async fn scope_isolation_for_receipts_and_audit() {
        let app = build_router(AppState::default());
        let _ = exec(&app, "acme", "x", r#"{"manifest":{},"vars":{}}"#, None).await;

        let list_y = Request::builder()
            .method("GET")
            .uri("/a/acme/t/y/v1/receipts")
            .header("authorization", "Bearer acme:y")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(list_y).await.unwrap();
        let v: Value =
            serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert!(v["keys"].as_array().unwrap().is_empty());

        let audit_y = Request::builder()
            .method("GET")
            .uri("/a/acme/t/y/v1/audit")
            .header("authorization", "Bearer acme:y")
            .body(Body::empty())
            .unwrap();
        let audit = app.oneshot(audit_y).await.unwrap();
        let a: Value =
            serde_json::from_slice(&audit.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert!(a["events"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn get_receipt_accepts_percent_encoded_cid_and_no_content_type() {
        let app = build_router(AppState::default());
        let (_, out, _) = exec(&app, "acme", "pro", r#"{"manifest":{},"vars":{}}"#, None).await;
        let cid = out["tip_cid"].as_str().unwrap().replace(':', "%3A");

        let get = Request::builder()
            .method("GET")
            .uri(format!("/a/acme/t/pro/v1/receipt/{cid}"))
            .header("authorization", "Bearer acme:pro")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(get).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn idempotency_replay_and_conflict() {
        let app = build_router(AppState::default());
        let first = exec(
            &app,
            "acme",
            "pro",
            r#"{"manifest":{},"vars":{"a":1}}"#,
            Some("same"),
        )
        .await;
        let replay = exec(
            &app,
            "acme",
            "pro",
            r#"{"manifest":{},"vars":{"a":1}}"#,
            Some("same"),
        )
        .await;
        assert_eq!(first.1["tip_cid"], replay.1["tip_cid"]);

        let conflict = exec(
            &app,
            "acme",
            "pro",
            r#"{"manifest":{},"vars":{"a":2}}"#,
            Some("same"),
        )
        .await;
        assert_eq!(conflict.0, StatusCode::BAD_REQUEST);
        assert_eq!(conflict.1["error"]["code"], "IDEMPOTENCY_CONFLICT");
        assert!(conflict.1["error"]["request_id"]
            .as_str()
            .unwrap()
            .starts_with("req_"));
    }

    #[tokio::test]
    async fn transition_endpoint_enforces_phase() {
        let app = build_router(AppState::default());
        let (_, exec_out, _) =
            exec(&app, "acme", "pro", r#"{"manifest":{},"vars":{}}"#, None).await;
        let wf_cid = exec_out["tip_cid"].as_str().unwrap();

        let wf_receipt = Request::builder()
            .method("GET")
            .uri(format!("/a/acme/t/pro/v1/receipt/{wf_cid}"))
            .header("authorization", "Bearer acme:pro")
            .body(Body::empty())
            .unwrap();
        let wf_resp = app.clone().oneshot(wf_receipt).await.unwrap();
        let wf_json: Value =
            serde_json::from_slice(&wf_resp.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        let tr_cid = wf_json["parent_cid"].as_str().unwrap();

        let tr = Request::builder()
            .method("GET")
            .uri(format!("/a/acme/t/pro/v1/transition/{tr_cid}"))
            .header("authorization", "Bearer acme:pro")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(tr).await.unwrap().status(),
            StatusCode::OK
        );

        let wrong = Request::builder()
            .method("GET")
            .uri(format!("/a/acme/t/pro/v1/transition/{wf_cid}"))
            .header("authorization", "Bearer acme:pro")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.oneshot(wrong).await.unwrap().status(),
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn policies_short_circuit() {
        let app = build_router(AppState::default());
        let (_, out, _) = exec(
            &app,
            "acme",
            "pro",
            r#"{"manifest":{"deny":true,"retry":true},"vars":{}}"#,
            None,
        )
        .await;

        assert_eq!(out["decision"], "DENY");
        let trace = out["policy_trace"].as_array().unwrap();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0]["id"], "deny-on-manifest:v1");
    }

    #[tokio::test]
    async fn cors_preflight_all_resolution_levels() {
        let app = build_router(AppState::default());

        let tenant_specific = Request::builder()
            .method("OPTIONS")
            .uri("/a/acme/t/pro/v1/execute")
            .header("origin", "https://tenant.acme.app")
            .header("access-control-request-method", "POST")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(tenant_specific).await.unwrap().status(),
            StatusCode::NO_CONTENT
        );

        let app_level = Request::builder()
            .method("OPTIONS")
            .uri("/a/acme/t/other/v1/execute")
            .header("origin", "https://acme.app")
            .header("access-control-request-method", "POST")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(app_level).await.unwrap().status(),
            StatusCode::NO_CONTENT
        );

        let global = Request::builder()
            .method("OPTIONS")
            .uri("/a/other/t/tenant/v1/execute")
            .header("origin", "https://global.app")
            .header("access-control-request-method", "POST")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.oneshot(global).await.unwrap().status(),
            StatusCode::NO_CONTENT
        );
    }

    #[tokio::test]
    async fn app_errors_include_request_id() {
        let app = build_router(AppState::default());

        let unauthorized = Request::builder()
            .method("POST")
            .uri("/a/acme/t/pro/v1/execute")
            .header("content-type", "application/json")
            .body(Body::from("{\"manifest\":{},\"vars\":{}}"))
            .unwrap();
        let u = app.clone().oneshot(unauthorized).await.unwrap();
        assert_eq!(u.status(), StatusCode::UNAUTHORIZED);
        let uv: Value =
            serde_json::from_slice(&u.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert!(uv["error"]["request_id"]
            .as_str()
            .unwrap()
            .starts_with("req_"));

        let forbidden = Request::builder()
            .method("GET")
            .uri("/a/acme/t/pro/v1/receipts")
            .header("authorization", "Bearer acme:wrong")
            .body(Body::empty())
            .unwrap();
        let f = app.clone().oneshot(forbidden).await.unwrap();
        assert_eq!(f.status(), StatusCode::FORBIDDEN);

        let not_found = Request::builder()
            .method("GET")
            .uri("/a/acme/t/pro/v1/receipt/b3:notfound")
            .header("authorization", "Bearer acme:pro")
            .body(Body::empty())
            .unwrap();
        let n = app.clone().oneshot(not_found).await.unwrap();
        assert_eq!(n.status(), StatusCode::NOT_FOUND);
        let nv: Value =
            serde_json::from_slice(&n.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert!(nv["error"]["request_id"]
            .as_str()
            .unwrap()
            .starts_with("req_"));
    }

    #[tokio::test]
    async fn rate_limit_scoped_and_headers_exposed() {
        let app = build_router(AppState::default());

        for _ in 0..RATE_LIMIT {
            let (status, _, headers) =
                exec(&app, "acme", "pro", r#"{"manifest":{},"vars":{}}"#, None).await;
            assert_eq!(status, StatusCode::OK);
            assert!(headers.get("x-ratelimit-limit").is_some());
            assert!(headers.get("x-ratelimit-remaining").is_some());
        }

        let limited = exec(&app, "acme", "pro", r#"{"manifest":{},"vars":{}}"#, None).await;
        assert_eq!(limited.0, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(limited.1["error"]["code"], "RATE_LIMITED");

        // different tenant should not share limiter bucket
        let other_tenant = exec(&app, "acme", "other", r#"{"manifest":{},"vars":{}}"#, None).await;
        assert_eq!(other_tenant.0, StatusCode::OK);
    }
}
