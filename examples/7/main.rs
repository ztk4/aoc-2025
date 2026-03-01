//! AoC Day 7
#![feature(result_flattening, macro_metavar_expr)]
use color_eyre::eyre::Result;
use libaoc::*;
use log::*;
use ocl::*;
use std::fs;

fn main() -> Result<()> {
  env_logger::init();
  color_eyre::install()?;

  let config = create_config!()?;
  info!(
    "Advent of Code day #{}, part {:?}!",
    config.day, config.part
  );

  let proque = ProQue::builder()
    .src(fs::read_to_string(config.local_dir.join("tachyon.cl"))?)
    .dims(1)
    .build()?;

  let input: Vec<_> = config
    .input
    .lines()
    .map(|line| Ok(line?.as_bytes().to_vec()))
    .collect::<Result<_>>()?;
  let height = input.len();
  let width = input[0].len();
  info!("Parsed {width}x{height} manifold");
  debug!("Manifold: {input:?}");

  let manifold = proque
    .buffer_builder::<u8>()
    .len(height * width)
    .copy_host_slice(&input.into_iter().flatten().collect::<Vec<_>>())
    .build()?;
  let vlookup = proque
    .buffer_builder::<u32>()
    .len(manifold.len())
    .fill_val(0)
    .build()?;

  let size: prm::Uint2 = [width as u32, height as u32].into();
  let build_vlookup = proque
    .kernel_builder("build_vertical_lookup")
    .global_work_size(config.group_size * width)
    .local_work_size(config.group_size)
    .arg(&manifold)
    .arg(size)
    .arg(&vlookup)
    .build()?;

  let propagate = proque
    .kernel_builder("propagate")
    .global_work_size(config.group_size)
    .local_work_size(config.group_size)
    .arg(&vlookup)
    .arg(size)
    .arg(&manifold)
    .arg_local::<u8>(manifold.len())
    .build()?;

  let mut count_partial = proque
    .kernel_builder("count_partial_reduce")
    .local_work_size(config.group_size)
    .arg(&manifold)
    .arg(manifold.len())
    .arg(b'*')
    .arg_named("partials", None::<&Buffer<u64>>)
    .build()?;

  let ngroups = get_mem_bound_reduce_num_groups_hint(&proque.device(), &count_partial)?;
  let gws = get_mem_bound_reduce_gws_hint(
    &proque.device(),
    &count_partial,
    config.group_size,
    manifold.len(),
  )?;

  let partials = proque
    .buffer_builder::<u64>()
    .len(ngroups)
    .fill_val(0)
    .build()?;
  let result = proque.buffer_builder::<u64>().fill_val(0).build()?;

  count_partial.set_default_global_work_size(gws);
  count_partial.set_arg("partials", &partials)?;

  let count_full = proque
    .kernel_builder("sum_full_reduce")
    .global_work_size(ngroups)
    .local_work_size(ngroups)
    .arg(&partials)
    .arg(&result)
    .build()?;

  let scratch = proque
    .buffer_builder::<u64>()
    .len(manifold.len())
    .fill_val(0)
    .build()?;
  let quantum_propagate = proque
    .kernel_builder("quantum_propagate")
    .global_work_size(config.group_size)
    .local_work_size(config.group_size)
    .arg(&vlookup)
    .arg(&manifold)
    .arg(size)
    .arg(&result)
    .arg(&scratch)
    .build()?;

  unsafe {
    build_vlookup.enq()?;
    debug!("VLookup: {:?}", buf2vec(&vlookup)?);

    match config.part {
      Part::One => {
        propagate.enq()?;
        debug!("Propagated Manifold: {:?}", buf2vec(&manifold)?);
        count_partial.enq()?;
        count_full.enq()?;
      }
      Part::Two => quantum_propagate.enq()?,
    }
  }

  println!(
    "Number of {}: {}",
    match config.part {
      Part::One => "Splits",
      Part::Two => "Timelines",
    },
    buf2vec(&result)?.into_iter().exactly_one()?
  );

  Ok(())
}
