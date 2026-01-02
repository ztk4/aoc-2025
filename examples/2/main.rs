//! AoC Day 2.
#![feature(result_flattening, macro_metavar_expr)]
use color_eyre::eyre::{Result, eyre};
use itertools::Itertools;
use libaoc::*;
use log::*;
use ocl::*;
use std::fs;

// Instead of a fixed 2-part sum (like in day 1),
// let's continually reduce with a fixed group size.
fn sum(
  buffer: &Buffer<u64>,
  ctx: &Context,
  dev: Device,
  prog: &Program,
  lws: usize,
) -> Result<u64> {
  let queue = Queue::new(ctx, dev, None)?;

  let mut partial_sums = buffer.clone();
  let mut reduce_kernels: Vec<Kernel> = vec![];
  while partial_sums.len() > 1 {
    let ngroups = (partial_sums.len() as f64 / lws as f64).ceil() as usize;
    let gws = ngroups * lws;
    let new_partials = Buffer::<u64>::builder()
      .queue(queue.clone())
      .len(ngroups)
      .fill_val(0)
      .build()?;
    reduce_kernels.push(
      Kernel::builder()
        .program(&prog)
        .name("sum")
        .queue(queue.clone())
        .global_work_size(gws)
        .local_work_size(lws)
        .arg(&partial_sums)
        .arg(partial_sums.len())
        .arg(&new_partials)
        .build()?,
    );
    partial_sums = new_partials;
  }

  unsafe {
    for kern in reduce_kernels {
      kern.enq()?;
    }
  }

  Ok(buf2vec(&partial_sums)?.into_iter().exactly_one()?)
}

/// Represents a (closed) range of ID values.
#[derive(Debug)]
struct IdRange {
  lo: usize,
  hi: usize,
}

impl IdRange {
  fn parse(input: impl AsRef<str>) -> Result<Self> {
    let (lo, hi) = input
      .as_ref()
      .split('-')
      .map(str::parse::<usize>)
      .collect_tuple()
      .ok_or_else(|| eyre!("Failed to parse {} as range", input.as_ref()))?;
    Ok(IdRange { lo: lo?, hi: hi? })
  }

  fn len(&self) -> usize {
    (self.hi - self.lo) + 1
  }

  /// Sums the invalid IDs in the range.
  fn sum_invalid(&self, ctx: &Context, dev: Device, prog: &Program, lws: usize) -> Result<u64> {
    let ngroups = (self.len() as f64 / lws as f64).ceil() as usize;
    let gws = ngroups * lws;
    let queue = Queue::new(ctx, dev, None)?;
    let partial_sums = Buffer::<u64>::builder()
      .queue(queue.clone())
      .len(ngroups)
      .fill_val(0)
      .build()?;

    // Generates sums of invalid IDs per local work group.
    let sum_invalid_partial = Kernel::builder()
      .program(&prog)
      .name("sum_invalid")
      .queue(queue.clone())
      .global_work_size(gws)
      .local_work_size(lws)
      .arg(self.lo)
      .arg(self.hi)
      .arg(&partial_sums)
      .build()?;

    unsafe {
      sum_invalid_partial.enq()?;
      debug!("{:?}", buf2vec(&partial_sums)?);
    }

    sum(&partial_sums, ctx, dev, prog, lws)
  }

  fn sum_invalid_extended(
    &self,
    ctx: &Context,
    dev: Device,
    prog: &Program,
    lws: usize,
  ) -> Result<u64> {
    let queue = Queue::new(ctx, dev, None)?;
    let invalid = Buffer::<u64>::builder()
      .queue(queue.clone())
      .len(self.len())
      .fill_val(0)
      .build()?;

    // All inputs are <16 digits long, so the largest chunks are half that.
    const MAX_CHUNK_SIZE: usize = 8;
    let get_invalid = Kernel::builder()
      .program(&prog)
      .name("get_invalid_extended")
      .queue(queue.clone())
      .global_work_size((self.len(), MAX_CHUNK_SIZE))
      .global_work_offset((0, 1)) // start with chunks of size 1 (not 0).
      .local_work_size((1, MAX_CHUNK_SIZE))
      .arg(self.lo)
      .arg(&invalid)
      .build()?;

    unsafe {
      get_invalid.enq()?;
      debug!("{:?}", buf2vec(&invalid)?);
    }

    sum(&invalid, ctx, dev, prog, lws)
  }
}

fn main() -> Result<()> {
  env_logger::init();
  color_eyre::install()?;

  let config = create_config!()?;
  info!(
    "Advent of Code day #{}, part {:?}!",
    config.day, config.part
  );
  let ranges: Vec<_> = config
    .input
    .lines()
    .exactly_one()??
    .split(',')
    .map(|range| IdRange::parse(range))
    .collect::<Result<_>>()?;

  let platform = Platform::default();
  let device = Device::first(platform)?;
  let context = Context::builder()
    .platform(platform)
    .devices(device.clone())
    .build()?;
  let program = Program::builder()
    .devices(device)
    .src(fs::read_to_string(config.local_dir.join("product_ids.cl"))?)
    .build(&context)?;
  let lws = config.group_size;

  let invalid_sums: Vec<_> = ranges
    .into_iter()
    .map(|range| match config.part {
      Part::One => range.sum_invalid(&context, device, &program, lws),
      Part::Two => range.sum_invalid_extended(&context, device, &program, lws),
    })
    .collect::<Result<_>>()?;
  info!("Invalid sums: {invalid_sums:?}");

  let queue = Queue::new(&context, device, None)?;
  let buffer = Buffer::<u64>::builder()
    .queue(queue.clone())
    .copy_host_slice(&invalid_sums)
    .len(invalid_sums.len())
    .build()?;
  println!(
    "Sum of invalid IDs: {}",
    sum(&buffer, &context, device, &program, lws)?
  );

  Ok(())
}
