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

const DIAL_LENGTH: u32 = 100;
const DIAL_INIT: u32 = 50;

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
      |line| -> Result<u32> {
        // Convert rotations to offsets mod dial length.
        Ok(match line.as_bytes()[0] {
          b'L' => (-line[1..].parse::<i32>()?).rem_euclid(DIAL_LENGTH as i32),
          b'R' => line[1..].parse::<i32>()?.rem_euclid(DIAL_LENGTH as i32),
          ch => return Err(eyre!("Invalid direction {ch}")),
        } as u32)
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
    .buffer_builder::<u32>()
    .copy_host_slice(&rots)
    .build()?;
  let group_tmp = proque
    .buffer_builder::<u32>()
    .len(ngroups)
    .fill_val(0)
    .build()?;
  // Performs cummulative sums within each group.
  let acc_local = proque
    .kernel_builder("accumulate_local")
    .global_work_size(global_size)
    .local_work_size(local_size)
    .arg(&buffer)
    .arg(buffer.len() as u32)
    .arg(DIAL_LENGTH as u32)
    .arg(&group_tmp)
    .arg_local::<u32>(local_size)
    .build()?;
  // Offsets group sums to create a global cummulative sum.
  let acc_global = proque
    .kernel_builder("accumulate_global")
    .global_work_size(global_size)
    .local_work_size(local_size)
    .arg(&buffer)
    .arg(buffer.len() as u32)
    .arg(DIAL_LENGTH as u32)
    .arg(&group_tmp)
    .arg_local::<u32>(ngroups)
    .build()?;
  // Counts the number of times the dial passes through 0 in groups.
  let count_zero = proque
    .kernel_builder("count")
    .global_work_size(global_size)
    .local_work_size(local_size)
    .arg(&buffer)
    .arg(buffer.len() as u32)
    .arg(0)
    .arg(&group_tmp)
    .arg_local::<u32>(local_size)
    .build()?;
  // Sums the counts in temp.
  let sum_counts = proque
    .kernel_builder("sum")
    .global_work_size(group_tmp.len())
    .local_work_size(group_tmp.len())
    .arg(&group_tmp)
    .arg_named("result", None::<&Buffer<u32>>)
    .arg_local::<u32>(group_tmp.len())
    .build()?;

  let result = proque.buffer_builder::<u32>().len(1).fill_val(0).build()?;
  sum_counts.set_arg("result", &result)?;
  unsafe {
    acc_local.enq()?;
    acc_global.enq()?;
    count_zero.enq()?;
    sum_counts.enq()?;
  }
  let mut count: u32 = 0;
  result.read(std::slice::from_mut(&mut count)).enq()?;
  println!("Password: {}", count);

  Ok(())
}
