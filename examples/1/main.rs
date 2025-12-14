//! Template example for starting a new day's challenge.
#![feature(result_flattening, macro_metavar_expr)]
use color_eyre::eyre::{Result, eyre};
use libaoc::*;
use log::*;
use ocl::*;
use std::fs;

#[derive(clap::Args)]
struct Options {
  /// Work group size for kernels.
  #[arg(long)]
  group_size: Option<usize>,
}

const DIAL_LENGTH: i64 = 100;
const DIAL_INIT: i64 = 50;

fn main() -> Result<()> {
  env_logger::init();
  color_eyre::install()?;

  let config = create_config!(challenge_args: Options)?;
  info!(
    "Advent of Code day #{}, part {:?}!",
    config.day, config.part
  );
  // Sample of using the parse macro, assuming each line is 3 ints.
  let rots: Vec<_> = std::iter::once(Ok(DIAL_INIT))
    .chain(parse_result_lines(
      config.input.lines(),
      |line| -> Result<i64> {
        // Convert rotations to offsets mod dial length.
        Ok(match line.as_bytes()[0] {
          b'L' => -line[1..].parse::<i64>()?,
          b'R' => line[1..].parse::<i64>()?,
          ch => return Err(eyre!("Invalid direction {ch}")),
        })
      },
    ))
    .collect::<Result<_>>()?;

  // Default to a group size of ~sqrt(N).
  let local_size = config
    .challenge_args
    .group_size
    .unwrap_or((rots.len() as f64).sqrt().ceil() as usize);
  let ngroups = (rots.len() as f64 / local_size as f64).ceil() as usize;
  let global_size = ngroups * local_size;

  let proque = ProQue::builder()
    .src(fs::read_to_string(config.local_dir.join("dial.cl"))?)
    .dims(rots.len())
    .build()?;
  let buffer = proque
    .buffer_builder::<i64>()
    .copy_host_slice(&rots)
    .build()?;
  let crossings = proque.buffer_builder::<i64>().fill_val(0).build()?;
  let is_nonneg = proque.buffer_builder::<i8>().fill_val(0).build()?;
  let group_tmp = proque
    .buffer_builder::<i64>()
    .len(ngroups)
    .fill_val(0)
    .build()?;
  // Breaks rotations down into euclid remainder, abs(quotient), and sign.
  let divmod = proque
    .kernel_builder("divmod_crossings")
    .global_work_size(global_size)
    .arg(&buffer)
    .arg(buffer.len())
    .arg(DIAL_LENGTH)
    .arg(&crossings)
    .arg(&is_nonneg)
    .build()?;
  // Performs cummulative sums within each group.
  let acc_local = proque
    .kernel_builder("accumulate_local")
    .global_work_size(global_size)
    .local_work_size(local_size)
    .arg(&buffer)
    .arg(buffer.len())
    .arg(DIAL_LENGTH)
    .arg(&group_tmp)
    .arg_local::<i64>(local_size)
    .build()?;
  // Offsets group sums to create a global cummulative sum.
  let acc_global = proque
    .kernel_builder("accumulate_global")
    .global_work_size(global_size)
    .local_work_size(local_size)
    .arg(&buffer)
    .arg(buffer.len())
    .arg(DIAL_LENGTH)
    .arg(&group_tmp)
    .arg_local::<i64>(ngroups)
    .build()?;
  // Counts the number of times the dial passes through 0 in groups.
  let count_zero = proque
    .kernel_builder("count")
    .global_work_size(global_size)
    .local_work_size(local_size)
    .arg(&buffer)
    .arg(buffer.len())
    .arg(0i64)
    .arg(&group_tmp)
    .arg_local::<i64>(local_size)
    .build()?;
  // Calculates the number of zero crossings per rotation.
  let get_xings = proque
    .kernel_builder("get_zero_crossings")
    .global_work_size(global_size)
    .arg(&buffer)
    .arg(buffer.len())
    .arg(DIAL_LENGTH)
    .arg(&is_nonneg)
    .arg(&crossings)
    .build()?;
  // Sums the counts of zero crossings per group.
  let sum_xings = proque
    .kernel_builder("sum")
    .global_work_size(global_size)
    .local_work_size(local_size)
    .arg(&crossings)
    .arg(crossings.len())
    .arg(&group_tmp)
    .arg_local::<i64>(local_size)
    .build()?;
  // Sums the counts in temp.
  let sum_counts = proque
    .kernel_builder("sum")
    .global_work_size(group_tmp.len())
    .local_work_size(group_tmp.len())
    .arg(&group_tmp)
    .arg(group_tmp.len())
    .arg_named("result", None::<&Buffer<i64>>)
    .arg_local::<i64>(group_tmp.len())
    .build()?;

  let result = proque.buffer_builder::<i64>().len(1).fill_val(0).build()?;
  sum_counts.set_arg("result", &result)?;
  unsafe {
    divmod.enq()?;
    acc_local.enq()?;
    acc_global.enq()?;

    match config.part {
      Part::One => count_zero.enq()?,
      Part::Two => {
        get_xings.enq()?;
        sum_xings.enq()?;
      }
    }

    sum_counts.enq()?;
  }
  println!("Password: {}", buf2vec(&result)?[0]);

  Ok(())
}
