mod app;

#[tokio::main]
async fn main() {
    let app = app::build_router(app::AppState::default());
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .expect("bind");
    axum::serve(listener, app).await.expect("server");
}
