//! Template example for starting a new day's challenge.
#![feature(result_flattening, macro_metavar_expr)]
use color_eyre::eyre::Result;
use libaoc::*;
use log::*;
use ocl::*;
use std::fs;

#[derive(clap::Args)]
struct Options {
  /// Whether to print OpenCL config info.
  #[arg(long, default_value_t = false)]
  show_config: bool,
}

fn main() -> Result<()> {
  env_logger::init();
  color_eyre::install()?;

  let config = create_config!(challenge_args: Options)?;
  info!(
    "Advent of Code day #{}, part {:?}!",
    config.day, config.part
  );
  // Sample of using the parse macro, assuming each line is 3 ints.
  let input: Vec<_> = parse_result_lines(config.input.lines(), |line| {
    vec_parse_tokens::<u64>(DelimitedTokens::by_whitespace(line))
  })
  .collect::<Result<_>>()?;
  println!("Input: {:?}", input);
  // Sample of using a singleton compute kernel to process inputs.
  // NOTE: This assumes input is a rectangular matrix => no handling for mixed length rows!
  let proque = ProQue::builder()
    .src(fs::read_to_string(config.local_dir.join("sum.cl"))?)
    .build()?;
  // Let's print some info about the defaults proque uses!
  if config.challenge_args.show_config {
    println!("Context: {:#}", proque.context());
    println!("Platform: {:#}", proque.context().platform()?.unwrap());
    println!(
      "Device 0: {:#}",
      proque.context().get_device_by_wrapping_index(0)
    );
  }

  // Let's reduce first by row, and then to a final sum.
  let nrows = input.len();
  let flat: Vec<_> = input.into_iter().flatten().collect();
  let flatbuf = Buffer::<u64>::builder()
    .queue(proque.queue().clone())
    .len(flat.len())
    .copy_host_slice(&flat)
    .build()?;
  let reduced = Buffer::<u64>::builder()
    .queue(proque.queue().clone())
    .len(nrows)
    .fill_val(0)
    .build()?;
  let result = Buffer::<u64>::builder()
    .queue(proque.queue().clone())
    .len(1)
    .fill_val(0)
    .build()?;
  // Pass 1
  let lws = flatbuf.len() / reduced.len();
  let kern = proque
    .kernel_builder("sum")
    .arg(&flatbuf)
    .arg(&reduced)
    .arg_local::<u64>(lws)
    .build()?;
  unsafe {
    kern
      .cmd()
      .global_work_size(flatbuf.len())
      .local_work_size(lws)
      .enq()?;
  }
  // Pass 2
  // NOTE: Obv inefficient to use a single work group... but it's a toy example.
  let kern = proque
    .kernel_builder("sum")
    .arg(&reduced)
    .arg(&result)
    .arg_local::<u64>(reduced.len())
    .build()?;
  unsafe {
    kern
      .cmd()
      .global_work_size(reduced.len())
      .local_work_size(reduced.len())
      .enq()?;
  }

  // We're using in-order execution above, so r/w to reduced and result should be syncrhonized.
  println!("Sum: {}", buf2vec(&result)?[0]);
  Ok(())
}
