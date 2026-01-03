//! AoC Day 3.
#![feature(result_flattening, macro_metavar_expr)]
use color_eyre::eyre::Result;
use libaoc::*;
use log::*;
use ocl::*;
use std::fs;

fn sum(buffer: &Buffer<u64>, proque: &ProQue, lws: usize) -> Result<u64> {
  let mut partial_sums = buffer.clone();
  let mut reduce_kerns: Vec<Kernel> = vec![];
  while partial_sums.len() > 1 {
    let ngroups = (partial_sums.len() as f64 / lws as f64).ceil() as usize;
    let new_partials = proque
      .buffer_builder::<u64>()
      .len(ngroups)
      .fill_val(0)
      .build()?;
    reduce_kerns.push(
      proque
        .kernel_builder("sum")
        .global_work_size(ngroups * lws)
        .local_work_size(lws)
        .arg(&partial_sums)
        .arg(partial_sums.len())
        .arg(&new_partials)
        .build()?,
    );
    partial_sums = new_partials;
  }

  unsafe {
    for kern in reduce_kerns {
      kern.enq()?
    }
  }

  Ok(buf2vec(&partial_sums)?.into_iter().exactly_one()?)
}

/// Represents a bank of batteries.
#[derive(Debug)]
struct Bank {
  jolts: Vec<u8>,
}

impl Bank {
  fn parse(line: impl AsRef<str>) -> Result<Self> {
    Ok(Self {
      jolts: vec_parse_tokens::<u8>(DelimitedTokens::each_char(line.as_ref()))?,
    })
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
  // Sample of using the parse macro, assuming each line is 3 ints.
  let banks: Vec<_> =
    parse_result_lines(config.input.lines(), |line| Bank::parse(line)).collect::<Result<_>>()?;
  let bank_size = banks[0].jolts.len(); // Assumes all banks are the same size.

  let proque = ProQue::builder()
    .src(fs::read_to_string(config.local_dir.join("batteries.cl"))?)
    .dims(banks.len() * bank_size)
    .build()?;

  // Copies batteries from each bank into a flat buffer.
  let batteries = proque
    .buffer_builder::<u64>()
    .copy_host_slice(
      &banks
        .iter()
        .map(|bank| bank.jolts.iter())
        .flatten()
        .map(|&b| b as u64)
        .collect::<Vec<_>>()
        .as_slice(),
    )
    .build()?;
  let joltages = proque
    .buffer_builder::<u64>()
    .len(banks.len())
    .fill_val(0)
    .build()?;
  // Finds the largest joltage possible for each bank.
  let find_largest = proque
    .kernel_builder("find_largest2")
    .global_work_size(batteries.len())
    .local_work_size(bank_size)
    .arg(&batteries)
    .arg(&joltages)
    .build()?;

  unsafe {
    find_largest.enq()?;
  }

  debug!("Largest joltages per bank: {:?}", buf2vec(&joltages)?);
  debug!("Debug output in `batteries` {:?}", buf2vec(&batteries)?);

  println!(
    "Total output joltage: {}",
    sum(&joltages, &proque, config.group_size)?
  );

  Ok(())
}
