//! AoC Day 5
#![feature(result_flattening, macro_metavar_expr)]
use color_eyre::eyre::Result;
use libaoc::*;
use log::*;
use ocl::*;
use std::fs;
use std::str::FromStr;

#[derive(Debug)]
struct Range {
  lo: u64,
  hi: u64,
}

impl FromStr for Range {
  type Err = color_eyre::eyre::Report;

  fn from_str(src: &str) -> Result<Self, Self::Err> {
    let (lo, hi) = tuple_parse_tokens!(DelimitedTokens::by_sep(src, "-") => (u64, u64))?;
    Ok(Range { lo, hi })
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

  let mut lines = config.input.lines();
  let ranges: Vec<_> = parse_result_lines(
    lines
      .by_ref()
      .take_while(|line| matches!(line, Ok(line) if !line.is_empty())),
    str::parse::<Range>,
  )
  .collect::<Result<_>>()?;
  let ids: Vec<_> = parse_result_lines(lines, str::parse::<u64>).collect::<Result<_>>()?;
  info!("Input Sizes: {}, {}", ranges.len(), ids.len());
  debug!("Input: {:?}, {:?}", ranges, ids);

  let proque = ProQue::builder()
    .src(fs::read_to_string(config.local_dir.join("cafe.cl"))?)
    .dims(1)
    .build()?;

  let ids = proque
    .buffer_builder::<u64>()
    .len(ids.len())
    .copy_host_slice(&ids)
    .build()?;
  let lo = proque
    .buffer_builder::<u64>()
    .len(ranges.len())
    .copy_host_slice(&ranges.iter().map(|r| r.lo).collect::<Vec<_>>())
    .build()?;
  let hi = proque
    .buffer_builder::<u64>()
    .len(ranges.len())
    .copy_host_slice(&ranges.into_iter().map(|r| r.hi).collect::<Vec<_>>())
    .build()?;

  let gs_lws = config.group_size;
  let mut get_counts = proque
    .kernel_builder("count_in_range_gs")
    .local_work_size(gs_lws)
    .arg(&ids)
    .arg(ids.len() as i32)
    .arg(&lo)
    .arg(&hi)
    .arg(lo.len() as i32)
    .arg_named("counts", None::<&Buffer<u64>>) // We don't know the size yet.
    .build()?;

  let r_lws = get_mem_bound_reduce_num_groups_hint(&proque.device(), &get_counts)?;
  let gws =
    get_mem_bound_reduce_gws_hint(&proque.device(), &get_counts, gs_lws, ids.len() * gs_lws)?;
  info!("GWS ({gws:?}), Reduce LWS ({r_lws})");

  let counts = proque
    .buffer_builder::<u64>()
    .len(r_lws)
    .fill_val(0)
    .build()?;
  let result = proque.buffer_builder::<u64>().fill_val(0).build()?;

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

  Ok(())
}
