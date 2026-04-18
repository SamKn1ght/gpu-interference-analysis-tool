use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = env!("CARGO_PKG_NAME"),
    version = env!("CARGO_PKG_VERSION")
)]
struct Args {
    /// Input .cu file
    #[arg()]
    input: String,
}

fn main() {
    let args = Args::parse();
    println!("{:?}", args);
}
