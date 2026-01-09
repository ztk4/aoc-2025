//! AoC Day 4
#![feature(result_flattening, macro_metavar_expr)]
use color_eyre::eyre::{Result, eyre};
use libaoc::*;
use log::*;
use ocl::{enums::*, *};
use std::fs;

trait SpatialDimsExt {
  fn to_padded_dims(&self, incr: impl Into<Self>) -> Result<Self>
  where
    Self: Sized;

  fn reduce(&self) -> Self;
}

impl SpatialDimsExt for SpatialDims {
  fn to_padded_dims(&self, incr: impl Into<Self>) -> Result<Self> {
    let i = incr.into();
    Ok(match (self.reduce(), i.reduce()) {
      (Self::One(x), Self::One(ix)) => util::padded_len(x, ix).into(),
      (Self::Two(x, y), Self::Two(ix, iy)) => {
        (util::padded_len(x, ix), util::padded_len(y, iy)).into()
      }
      (Self::Three(x, y, z), Self::Three(ix, iy, iz)) => (
        util::padded_len(x, ix),
        util::padded_len(y, iy),
        util::padded_len(z, iz),
      )
        .into(),
      _ => {
        return Err(eyre!(
          "Can't pad dims of cardinality {} to target of cardinality {}: {self:?} -> {i:?}",
          self.dim_count(),
          i.dim_count()
        ));
      }
    })
  }

  fn reduce(&self) -> SpatialDims {
    match *self {
      Self::Two(x, 1) | Self::Three(x, 1, 1) => Self::One(x),
      Self::Three(x, y, 1) => Self::Two(x, y),
      s => s,
    }
  }
}

fn sum(buffer: &Buffer<i64>, proque: &ProQue, lws: usize) -> Result<i64> {
  let mut partial_sums = buffer.clone();
  let mut reduce_kerns: Vec<Kernel> = vec![];
  while partial_sums.len() > 1 {
    let gws = util::padded_len(partial_sums.len(), lws);
    let new_partials = proque
      .buffer_builder::<i64>()
      .len(gws / lws)
      .fill_val(0)
      .build()?;
    reduce_kerns.push(
      proque
        .kernel_builder("sum")
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
    for kern in reduce_kerns {
      kern.enq()?;
    }
  }

  Ok(buf2vec(&partial_sums)?.into_iter().exactly_one()?)
}

fn main() -> Result<()> {
  env_logger::init();
  color_eyre::install()?;

  let config = create_config!()?;
  info!(
    "Advent of Code day #{}, part {:?}!",
    config.day, config.part
  );
  let input: Vec<_> = parse_result_lines(config.input.lines(), |line| {
    vec_parse_tokens::<char>(DelimitedTokens::each_char(line))
  })
  .collect::<Result<_>>()?;
  let proque = ProQue::builder()
    .src(fs::read_to_string(config.local_dir.join("printroom.cl"))?)
    .dims(1) // unused
    .build()?;

  // Modeling the map as a 2D image.
  let map = Image::<i8>::builder()
    .queue(proque.queue().clone())
    .flags(flags::MEM_READ_ONLY)
    .image_type(MemObjectType::Image2d)
    .dims((input[0].len(), input.len())) // assume rectangular
    .channel_order(ImageChannelOrder::Luminance) // single channel
    .channel_data_type(ImageChannelDataType::SignedInt8)
    .copy_host_slice(
      &input
        .into_iter()
        .map(|row| row.into_iter())
        .flatten()
        .map(|ch| match ch {
          '.' => 0,
          ch => ch as i8,
        })
        .collect::<Vec<_>>(),
    )
    .build()?;
  let accessible = proque
    .buffer_builder::<i64>()
    .len(map.element_count())
    .fill_val(0)
    .build()?;

  // Hardcoding a reasonably efficient working size for now.
  let lws = (32, 32);
  let find_accessible = proque
    .kernel_builder("find_accessible")
    .global_work_size(map.dims().to_padded_dims(lws)?)
    .local_work_size(lws)
    .arg(&map)
    .arg(4)
    .arg(&accessible)
    .build()?;

  unsafe {
    find_accessible.enq()?;
  }

  debug!("Map: {:?}", img2vec(&map)?);
  debug!("Accessible: {:?}", buf2vec(&accessible)?);

  println!("Sum: {}", sum(&accessible, &proque, config.group_size)?);
  Ok(())
}
