use std::path::{Path, PathBuf};

const DEFAULT_CONFIG_FILE_PATH: &str = "giat_config";

#[derive(Debug)]
pub struct Config {
    input_file_path: PathBuf,
    config_file_path: PathBuf,
    output_dir: PathBuf,
}
impl Config {
    pub fn get_config_file_path(&self) -> &Path {
        &self.config_file_path
    }
    pub fn get_output_dir(&self) -> &Path {
        &self.output_dir
    }
    pub fn new_output_file(&self, path: impl AsRef<Path>) -> PathBuf {
        self.output_dir.clone().join(path)
    }
}

pub struct ConfigBuilder {
    input_file_path: Option<PathBuf>,
    config_file_path: Option<PathBuf>,
    output_dir: Option<PathBuf>,
}
impl ConfigBuilder {
    pub fn new() -> ConfigBuilder {
        ConfigBuilder {
            input_file_path: None,
            config_file_path: None,
            output_dir: None,
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
            output_dir: self
                .output_dir
                .unwrap_or_else(|| PathBuf::from(Self::default_output_dir())),
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

    fn default_output_dir() -> String {
        chrono::Local::now().format("results_%Y%m%d-%H%M%S").to_string()
    }
}
