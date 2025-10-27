use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{Mutex, OnceCell};
use tonic::transport::{Channel, Endpoint};

use proto::video_analyzer_client::VideoAnalyzerClient;

pub mod proto {
    tonic::include_proto!("videoanalyzer");
}

pub struct GrpcState {
    endpoint: String,
    client: OnceCell<Arc<Mutex<VideoAnalyzerClient<Channel>>>>,
}

impl GrpcState {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            client: OnceCell::new(),
        }
    }

    pub async fn get_client(&self) -> Result<Arc<Mutex<VideoAnalyzerClient<Channel>>>> {
        let endpoint = self.endpoint.clone();
        let client = self
            .client
            .get_or_try_init(|| async {
                let channel = Endpoint::from_shared(endpoint.clone())?
                    .connect()
                    .await?;
                Ok(Arc::new(Mutex::new(VideoAnalyzerClient::new(channel))))
            })
            .await?;

        Ok(client.clone())
    }
}
