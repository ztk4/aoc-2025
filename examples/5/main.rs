//! AoC Day 5
#![feature(result_flattening, macro_metavar_expr)]
use color_eyre::eyre::{Result, eyre};
use libaoc::*;
use log::*;
use ocl::*;
use std::fs;
use std::str::FromStr;

#[derive(Debug)]
struct Range {
  lo: i64,
  hi: i64,
}

impl FromStr for Range {
  type Err = color_eyre::eyre::Report;

  fn from_str(src: &str) -> Result<Self, Self::Err> {
    let (lo, hi) = tuple_parse_tokens!(DelimitedTokens::by_sep(src, "-") => (i64, i64))?;
    Ok(Range { lo, hi })
  }
}

/// Sorts buffer using bitonic sort.
/// NB: buffer's length and group_size must be a power of two.
fn bitonic_sort(buffer: &Buffer<prm::Long2>, proque: &ProQue, group_size: usize) -> Result<()> {
  if !buffer.len().is_power_of_two() {
    return Err(eyre!(
      "Buffer of length {} must be a power of two",
      buffer.len()
    ));
  }
  if !group_size.is_power_of_two() {
    return Err(eyre!("Group size {group_size} must be a power of two"));
  }

  let lws = std::cmp::min(group_size, buffer.len() >> 1);
  let mut k = 2;
  let mut kerns: Vec<Kernel> = vec![];
  while k <= buffer.len() {
    let scratch_size = std::cmp::max(lws << 1, k);
    kerns.push(
      proque
        .kernel_builder("bitonic_merge")
        .global_work_size(buffer.len() * lws / scratch_size)
        .local_work_size(lws)
        .arg(buffer)
        .arg(k as i32)
        .arg_local::<prm::Long2>(scratch_size)
        .arg(scratch_size as i32)
        .build()?,
    );

    k <<= 1;
  }

  for kern in kerns {
    unsafe { kern.enq()? }
    debug!("buffer: {:?}", buf2vec(buffer)?);
  }

  Ok(())
}

fn main() -> Result<()> {
  env_logger::init();
  color_eyre::install()?;

  let config = create_config!()?;
  info!(
    "Advent of Code day #{}, part {:?}!",
    config.day, config.part
  );

  let mut lines = config.input.lines();
  let ranges: Vec<_> = parse_result_lines(
    lines
      .by_ref()
      .take_while(|line| matches!(line, Ok(line) if !line.is_empty())),
    str::parse::<Range>,
  )
  .collect::<Result<_>>()?;
  let ids: Vec<_> = parse_result_lines(lines, str::parse::<i64>).collect::<Result<_>>()?;
  info!("Input Sizes: {}, {}", ranges.len(), ids.len());
  debug!("Input: {:?}, {:?}", ranges, ids);

  let proque = ProQue::builder()
    .src(fs::read_to_string(config.local_dir.join("cafe.cl"))?)
    .dims(1)
    .build()?;

  match config.part {
    Part::One => {
      let ids = proque
        .buffer_builder::<i64>()
        .len(ids.len())
        .copy_host_slice(&ids)
        .build()?;
      let ranges = proque
        .buffer_builder::<prm::Long2>()
        .len(ranges.len())
        .copy_host_slice(
          &ranges
            .iter()
            .map(|r| [r.lo, r.hi].into())
            .collect::<Vec<_>>(),
        )
        .build()?;

      let gs_lws = config.group_size;
      let mut get_counts = proque
        .kernel_builder("count_in_range_gs")
        .local_work_size(gs_lws)
        .arg(&ids)
        .arg(ids.len() as i32)
        .arg(&ranges)
        .arg(ranges.len() as i32)
        .arg_named("counts", None::<&Buffer<i64>>) // We don't know the size yet.
        .build()?;

      let r_lws = get_mem_bound_reduce_num_groups_hint(&proque.device(), &get_counts)?;
      let gws =
        get_mem_bound_reduce_gws_hint(&proque.device(), &get_counts, gs_lws, ids.len() * gs_lws)?;
      info!("GWS ({gws:?}), Reduce LWS ({r_lws})");

      let counts = proque
        .buffer_builder::<i64>()
        .len(r_lws)
        .fill_val(0)
        .build()?;
      let result = proque.buffer_builder::<i64>().fill_val(0).build()?;

      get_counts.set_default_global_work_size(gws);
      get_counts.set_arg("counts", Some(&counts))?;

      let reduce_counts = proque
        .kernel_builder("reduce_counts")
        .global_work_size(r_lws)
        .local_work_size(r_lws)
        .arg(&counts)
        .arg(&result)
        .build()?;

      unsafe {
        get_counts.enq()?;
        reduce_counts.enq()?;
      }

      println!(
        "Fresh Count: {}",
        buf2vec(&result)?.into_iter().exactly_one()?
      );
    }
    Part::Two => {
      let size = (ranges.len() << 1).next_power_of_two();
      let endpoints = proque
        .buffer_builder::<prm::Long2>()
        .len(size)
        .copy_host_slice(
          &ranges
            .iter()
            // Using `1` to denote opening endpoints, and `-1` for closing.
            .map(|r| [r.lo, 1])
            .chain(ranges.iter().map(|r| [r.hi, -1]))
            // Sentinel values to pad to power of two.
            .chain(std::iter::repeat([0, 0]))
            .take(size)
            .map_into::<prm::Long2>()
            .collect::<Vec<_>>(),
        )
        .build()?;

      bitonic_sort(&endpoints, &proque, config.group_size)?;
    }
  }

  Ok(())
}
