use std::path::PathBuf;

const DEFAULT_CONFIG_FILE_PATH: &str = "giat_config";

#[derive(Debug)]
pub struct Config {
    input_file_path: PathBuf,
    config_file_path: PathBuf,
}

pub struct ConfigBuilder {
    input_file_path: Option<PathBuf>,
    config_file_path: Option<PathBuf>,
}
impl ConfigBuilder {
    pub fn new() -> ConfigBuilder {
        ConfigBuilder {
            input_file_path: None,
            config_file_path: None,
        }
    }
    pub fn build(self) -> Option<Config> {
        if self.input_file_path.is_none() {
            return None;
        }
        Some(Config {
            input_file_path: self.input_file_path.unwrap(),
            config_file_path: self
                .config_file_path
                .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG_FILE_PATH)),
        })
    }

    pub fn input_file_path(&mut self, path: impl Into<PathBuf>) -> &mut Self {
        self.input_file_path = Some(path.into());
        self
    }
    pub fn config_file_path(&mut self, path: impl Into<PathBuf>) -> &mut Self {
        self.config_file_path = Some(path.into());
        self
    }
}
