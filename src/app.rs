use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{Path, State},
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Wa,
    Transition,
    Wf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
        (status, Json(self)).into_response()
    }
}

#[derive(Clone)]
pub struct AppState {
    ledger: Arc<Mutex<HashMap<String, ReceiptV1>>>,
    events: Arc<Mutex<Vec<String>>>,
    cors: Arc<CorsConfig>,
    idempotency: Arc<Mutex<HashMap<String, (String, Value)>>>,
    rate: Arc<Mutex<HashMap<String, VecDeque<Instant>>>>,
}

impl Default for AppState {
    fn default() -> Self {
        let mut app_allowed = HashMap::new();
        app_allowed.insert("acme".into(), vec!["https://acme.app".into()]);
        Self {
            ledger: Arc::new(Mutex::new(HashMap::new())),
            events: Arc::new(Mutex::new(vec![])),
            cors: Arc::new(CorsConfig {
                global_allowed: vec!["https://global.app".into()],
                app_allowed,
                app_tenant_allowed: HashMap::new(),
            }),
            idempotency: Arc::new(Mutex::new(HashMap::new())),
            rate: Arc::new(Mutex::new(HashMap::new())),
        }
    }
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
        .route("/v1/execute", post(execute_legacy))
        .route("/v1/receipt/:cid", get(get_receipt_legacy))
        .route("/v1/receipts", get(list_receipts_legacy))
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
    let request_id = format!("req_{}", blake3::hash(req.uri().path().as_bytes()).to_hex());
    let scope = parse_scope_from_path(req.uri().path());
    if let Some(origin) = req.headers().get("origin").and_then(|v| v.to_str().ok()) {
        if !state.cors.is_origin_allowed(origin, scope.as_ref()) {
            return AppErrorEnvelope {
                error: AppError::new("CORS_DENIED", "origin is not allowed", request_id, 0),
            }
            .into_response();
        }
        if req.method() == Method::OPTIONS {
            let mut r = Response::new(Body::empty());
            *r.status_mut() = StatusCode::NO_CONTENT;
            add_cors_headers(r.headers_mut(), origin);
            return r;
        }
    }
    if req.method() != Method::OPTIONS {
        if let Err(err) = check_auth(req.headers(), scope.as_ref(), request_id.clone()) {
            return err.into_response();
        }
    }

    if let Some(sc) = scope {
        let client_id = extract_client(req.headers()).unwrap_or_else(|| "anonymous".into());
        let rl_key = format!(
            "rl:{}:{}:{}:{}:{}",
            sc.app,
            sc.tenant,
            client_id,
            req.method(),
            req.uri().path()
        );
        if let Some(retry_after) = apply_rate_limit(&state, rl_key) {
            return AppErrorEnvelope {
                error: AppError::new("RATE_LIMITED", "too many requests", request_id, retry_after),
            }
            .into_response();
        }
        req.extensions_mut().insert(sc);
    }

    let response = next.run(req).await;
    if let Some(origin) = response.headers().get("access-control-allow-origin") {
        let _ = origin;
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
        HeaderValue::from_static("x-ratelimit-limit,x-ratelimit-remaining,retry-after"),
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

fn apply_rate_limit(state: &AppState, key: String) -> Option<u64> {
    let mut rate = state.rate.lock().unwrap();
    let now = Instant::now();
    let window = Duration::from_secs(60);
    let bucket = rate.entry(key).or_default();
    while bucket
        .front()
        .is_some_and(|t| now.duration_since(*t) > window)
    {
        bucket.pop_front();
    }
    if bucket.len() >= 20 {
        return Some(60);
    }
    bucket.push_back(now);
    None
}

async fn execute_scoped(
    Path((app, tenant)): Path<(String, String)>,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<ExecuteRequest>,
) -> Result<Json<ExecuteResponse>, AppErrorEnvelope> {
    execute_inner(Scope::new(app, tenant), state, headers, payload).await
}

async fn execute_legacy(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<ExecuteRequest>,
) -> Result<Json<ExecuteResponse>, AppErrorEnvelope> {
    execute_inner(
        Scope::new(DEFAULT_APP, DEFAULT_TENANT),
        state,
        headers,
        payload,
    )
    .await
}

async fn execute_inner(
    scope: Scope,
    state: AppState,
    headers: HeaderMap,
    payload: ExecuteRequest,
) -> Result<Json<ExecuteResponse>, AppErrorEnvelope> {
    let idemp_key = headers
        .get("idempotency-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("none");
    let body = canonical_input(
        &payload.manifest,
        &payload.vars,
        payload
            .options
            .as_ref()
            .and_then(|o| o.ghost)
            .unwrap_or(false),
    );
    let body_sha = blake3::hash(body.as_bytes()).to_hex().to_string();
    let idem_key = format!(
        "idem:{}:{}:POST:execute:{}",
        scope.app, scope.tenant, idemp_key
    );
    if let Some(existing) = state.idempotency.lock().unwrap().get(&idem_key).cloned() {
        if existing.0 == body_sha {
            return Ok(Json(serde_json::from_value(existing.1).unwrap()));
        }
    }

    let wa_cid = cid_from_bytes(format!("{}:wa", body).as_bytes());
    let tr_cid = cid_from_bytes(format!("{}:{}:tr", body, wa_cid).as_bytes());
    let decision = decide(&payload.manifest);
    let wf_cid = cid_from_bytes(format!("{}:{}:{:?}:wf", body, tr_cid, decision).as_bytes());

    let runtime = RuntimeInfo {
        did: "did:key:z6MkRuntime".to_string(),
        sha256: "runtime-sha256".to_string(),
    };

    let receipts = vec![
        mk_receipt(scope.clone(), Phase::Wa, wa_cid.clone(), None, None),
        mk_receipt(
            scope.clone(),
            Phase::Transition,
            tr_cid.clone(),
            Some(wa_cid.clone()),
            None,
        ),
        mk_receipt(
            scope.clone(),
            Phase::Wf,
            wf_cid.clone(),
            Some(tr_cid.clone()),
            Some(decision.clone()),
        ),
    ];

    {
        let mut ledger = state.ledger.lock().unwrap();
        for rec in receipts {
            ledger.insert(format!("{}{}", scope.key_prefix(), rec.body_cid), rec);
        }
    }
    state.events.lock().unwrap().push(wf_cid.clone());

    let response = ExecuteResponse {
        decision: decision.clone(),
        tip_cid: wf_cid.clone(),
        links: HashMap::from([
            (
                "self".into(),
                format!("/a/{}/t/{}/v1/receipt/{}", scope.app, scope.tenant, wf_cid),
            ),
            (
                "verify".into(),
                format!("/a/{}/t/{}/v1/verify/{}", scope.app, scope.tenant, wf_cid),
            ),
        ]),
        runtime,
        policy_trace: vec![PolicyTrace {
            id: "allow-all:v1".into(),
            decision,
        }],
    };

    state.idempotency.lock().unwrap().insert(
        idem_key,
        (body_sha, serde_json::to_value(&response).unwrap()),
    );
    Ok(Json(response))
}

fn canonical_input(manifest: &Value, vars: &Value, ghost: bool) -> String {
    let mut m = serde_json::Map::new();
    m.insert("manifest".into(), manifest.clone());
    m.insert("vars".into(), vars.clone());
    m.insert("ghost".into(), Value::Bool(ghost));
    serde_json::to_string(&Value::Object(m)).unwrap()
}

fn decide(manifest: &Value) -> Decision {
    if manifest
        .get("deny")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Decision::Deny;
    }
    if manifest
        .get("retry")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Decision::Retry;
    }
    Decision::Pass
}

fn mk_receipt(
    scope: Scope,
    phase: Phase,
    cid: String,
    parent: Option<String>,
    decision: Option<Decision>,
) -> ReceiptV1 {
    let payload_hex = hex::encode(cid.as_bytes());
    ReceiptV1 {
        scope,
        phase,
        body_cid: cid,
        runtime_did: "did:key:z6MkRuntime".into(),
        runtime_sha256: Some("runtime-sha256".into()),
        jws_alg: "Ed25519".into(),
        jws_kid: "did:key:z6MkRuntime".into(),
        jws_payload_hex: payload_hex,
        parent_cid: parent,
        decision,
        ts_ms: Utc::now().timestamp_millis(),
    }
}

fn cid_from_bytes(bytes: &[u8]) -> String {
    format!("b3:{}", blake3::hash(bytes).to_hex())
}

async fn get_receipt_scoped(
    Path((app, tenant, cid)): Path<(String, String, String)>,
    State(state): State<AppState>,
) -> Result<Json<ReceiptV1>, AppErrorEnvelope> {
    get_receipt_inner(Scope::new(app, tenant), cid, state)
}

async fn get_receipt_legacy(
    Path(cid): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<ReceiptV1>, AppErrorEnvelope> {
    get_receipt_inner(Scope::new(DEFAULT_APP, DEFAULT_TENANT), cid, state)
}

fn get_receipt_inner(
    scope: Scope,
    cid: String,
    state: AppState,
) -> Result<Json<ReceiptV1>, AppErrorEnvelope> {
    let cid = cid.replace("%3A", ":");
    let key = format!("{}{}", scope.key_prefix(), cid);
    let ledger = state.ledger.lock().unwrap();
    let Some(r) = ledger.get(&key) else {
        return Err(AppErrorEnvelope {
            error: AppError::new("NOT_FOUND", "receipt not found", "req_notfound".into(), 0),
        });
    };
    Ok(Json(r.clone()))
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
        .cloned()
        .collect();
    Json(json!({ "keys": keys }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn deterministic_execute() {
        let app = build_router(AppState::default());
        let body = r#"{"manifest":{},"vars":{"doc":"x"},"options":{"ghost":true}}"#;
        let req = || {
            Request::builder()
                .method("POST")
                .uri("/a/acme/t/pro/v1/execute")
                .header("authorization", "Bearer acme:pro")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap()
        };
        let r1 = app.clone().oneshot(req()).await.unwrap();
        let b1 = r1.into_body().collect().await.unwrap().to_bytes();
        let v1: Value = serde_json::from_slice(&b1).unwrap();
        let r2 = app.oneshot(req()).await.unwrap();
        let b2 = r2.into_body().collect().await.unwrap().to_bytes();
        let v2: Value = serde_json::from_slice(&b2).unwrap();
        assert_eq!(v1["tip_cid"], v2["tip_cid"]);
    }

    #[tokio::test]
    async fn scope_isolated() {
        let app = build_router(AppState::default());
        let do_exec = |tenant| {
            Request::builder()
                .method("POST")
                .uri(format!("/a/acme/t/{tenant}/v1/execute"))
                .header("authorization", format!("Bearer acme:{tenant}"))
                .header("content-type", "application/json")
                .body(Body::from("{\"manifest\":{},\"vars\":{}}"))
                .unwrap()
        };
        let _ = app.clone().oneshot(do_exec("x")).await.unwrap();
        let list_y = Request::builder()
            .method("GET")
            .uri("/a/acme/t/y/v1/receipts")
            .header("authorization", "Bearer acme:y")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(list_y).await.unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["keys"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn cors_preflight_allowed() {
        let app = build_router(AppState::default());
        let req = Request::builder()
            .method("OPTIONS")
            .uri("/a/acme/t/pro/v1/execute")
            .header("origin", "https://acme.app")
            .header("access-control-request-method", "POST")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    }
}
