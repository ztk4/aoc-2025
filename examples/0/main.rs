//! Template example for starting a new day's challenge.
#![feature(result_flattening, macro_metavar_expr)]
use color_eyre::eyre::Result;
use libaoc::*;
use log::*;

fn main() -> Result<()> {
  env_logger::init();
  color_eyre::install()?;

  let config = create_config!()?;
  info!(
    "Advent of Code day #{}, part {:?}!",
    config.day, config.part
  );
  // Sample of using the parse macro, assuming each line is 3 ints.
  println!(
    "Input: {:?}",
    parse_result_lines(config.input.lines(), |line| tuple_parse_tokens!(
      DelimitedTokens::by_whitespace(line) => (u64, u64, u64)
    ))
    .collect::<Result<Vec<_>>>()?
  );

  Ok(())
}
