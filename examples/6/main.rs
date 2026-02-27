//! Aoc Day 6
#![feature(result_flattening, macro_metavar_expr)]
use color_eyre::eyre::{Result, eyre};
use libaoc::*;
use log::*;
use ocl::*;
use std::fs;

fn main() -> Result<()> {
  env_logger::init();
  color_eyre::install()?;

  let mut config = create_config!()?;
  info!(
    "Advent of Code day #{}, part {:?}!",
    config.day, config.part
  );

  let proque = ProQue::builder()
    .src(fs::read_to_string(config.local_dir.join("math.cl"))?)
    .dims(1)
    .build()?;

  match config.part {
    Part::One => {
      let input: Vec<_> = parse_result_lines(config.input.lines(), |line| {
        match line.trim().chars().next().unwrap() {
          '0'..='9' => vec_parse_tokens::<u64>(DelimitedTokens::by_whitespace(line)),
          // For the operations, let's just map to code points.
          '*' | '+' => vec_parse_tokens::<char>(DelimitedTokens::by_whitespace(line))
            .map(|v| v.into_iter().map(|c| c as u64).collect()),
          c => Err(eyre!("Unsupported char '{c}'")),
        }
      })
      .collect::<Result<_>>()?;
      let nprob = input[0].len();
      info!("Found {} lines encoding {nprob} problems", input.len());
      debug!("Input: {:?}", input);

      let equations = proque
        .buffer_builder::<u64>()
        .len(input.len() * nprob)
        .copy_host_slice(&input.iter().cloned().flatten().collect::<Vec<_>>())
        .build()?;

      let mut sum_equations_partial = proque
        .kernel_builder("sum_equations_partial_reduce")
        .local_work_size(config.group_size)
        .arg(&equations)
        .arg(nprob)
        .arg(input.len())
        .arg_named("partials", None::<&Buffer<u64>>)
        .build()?;

      let ngroups = get_mem_bound_reduce_num_groups_hint(&proque.device(), &sum_equations_partial)?;
      let gws = get_mem_bound_reduce_gws_hint(
        &proque.device(),
        &sum_equations_partial,
        config.group_size,
        nprob,
      )?;

      let partials = proque
        .buffer_builder::<u64>()
        .len(ngroups)
        .fill_val(0)
        .build()?;
      let result = proque.buffer_builder::<u64>().fill_val(0).build()?;

      sum_equations_partial.set_default_global_work_size(gws);
      sum_equations_partial.set_arg("partials", &partials)?;

      let sum_full_reduce = proque
        .kernel_builder("sum_full_reduce")
        .global_work_size(ngroups)
        .local_work_size(ngroups)
        .arg(&partials)
        .arg(&result)
        .build()?;

      unsafe {
        sum_equations_partial.enq()?;
        debug!("Partials: {:?}", buf2vec(&partials)?);
        sum_full_reduce.enq()?;
      }

      println!(
        "Equation Sum: {}",
        buf2vec(&result)?.into_iter().exactly_one()?
      );
    }
    Part::Two => {
      let mut numchars: Vec<u8> = vec![];
      let mut operators: Vec<u8> = vec![];
      loop {
        let mut line: Vec<u8> = vec![];
        if config.input.read_until(b'\n', &mut line)? == 0 {
          break;
        }
        match line[0] {
          b'+' | b'*' => operators = line,
          _ => numchars.append(&mut line),
        };
      }
      // Assuming every line is the same length for now.
      let llen = operators.len();
      let nlines = numchars.len() / llen;
      info!("Found {} lines", nlines + 1);
      debug!("Input: {numchars:?}{operators:?}");

      let chars = proque
        .buffer_builder::<u8>()
        .len(numchars.len())
        .copy_host_slice(&numchars)
        .build()?;
      let ops = proque
        .buffer_builder::<u8>()
        .len(operators.len())
        .copy_host_slice(&operators)
        .build()?;

      let numbers = proque
        .buffer_builder::<u64>()
        .len(llen)
        .fill_val(0)
        .build()?;

      let parse_numbers = proque
        .kernel_builder("parse_numbers")
        .global_work_size(llen.next_multiple_of(config.group_size))
        .local_work_size(config.group_size)
        .arg(&chars)
        .arg(nlines)
        .arg(llen)
        .arg(&numbers)
        .build()?;

      let mut sum_equations_partial = proque
        .kernel_builder("sum_col_equations_partial_reduce")
        .local_work_size(config.group_size)
        .arg(&numbers)
        .arg(&ops)
        .arg(llen)
        .arg_named("partials", None::<&Buffer<u64>>)
        .build()?;

      let ngroups = get_mem_bound_reduce_num_groups_hint(&proque.device(), &sum_equations_partial)?;
      let gws = get_mem_bound_reduce_gws_hint(
        &proque.device(),
        &sum_equations_partial,
        config.group_size,
        llen,
      )?;

      let partials = proque
        .buffer_builder::<u64>()
        .len(ngroups)
        .fill_val(0)
        .build()?;
      let result = proque.buffer_builder::<u64>().fill_val(0).build()?;

      sum_equations_partial.set_default_global_work_size(gws);
      sum_equations_partial.set_arg("partials", &partials)?;

      let sum_full_reduce = proque
        .kernel_builder("sum_full_reduce")
        .global_work_size(ngroups)
        .local_work_size(ngroups)
        .arg(&partials)
        .arg(&result)
        .build()?;

      unsafe {
        parse_numbers.enq()?;
        debug!("Numbers: {:?}", buf2vec(&numbers)?);
        sum_equations_partial.enq()?;
        debug!("Partials: {:?}", buf2vec(&partials)?);
        sum_full_reduce.enq()?;
      }

      println!(
        "Equation Sum: {}",
        buf2vec(&result)?.into_iter().exactly_one()?
      );
    }
  }

  Ok(())
}
