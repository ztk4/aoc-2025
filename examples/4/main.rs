//! AoC Day 4
#![feature(result_flattening, macro_metavar_expr)]
use color_eyre::eyre::{Result, eyre};
use ilog::IntLog;
use libaoc::*;
use log::*;
use num_traits::Num;
use ocl::{enums::*, *};
use std::{fs, ops};

trait IntLogExt {
  fn prev_power_of_two(self) -> Self;
}

impl<T: IntLog + Num + ops::Shl<usize, Output = T>> IntLogExt for T {
  // NOTE: Panics if self < 0.
  fn prev_power_of_two(self) -> Self {
    if self == Self::zero() {
      self
    } else {
      Self::one() << self.log2()
    }
  }
}

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

// Returns an heuristically optimal number of groups to use when scheduling a memory-bound kernel.
// The logic is that we would want a small multiple of #CU s.t. latency hiding "kicks in".
// Since my kernels (for AoC) tend to be overwhelmingly memory bound, I'm opting for 7x.
fn get_mem_bound_num_groups_hint(device: &Device) -> Result<usize> {
  let DeviceInfoResult::MaxComputeUnits(cu) = device.info(DeviceInfo::MaxComputeUnits)? else {
    panic!("Wrong device info type!");
  };

  Ok(cu as usize * 7)
}

// Similar to the above, but tuned for a 2-pass reduce operation.
// In this case, we want the result to be a multiple of the "warp size", and less than the max group size.
// NOTE: Usually the warp size will be a power of 2, and there can be advantages to picking another
// power of 2 as opposed to just any multiple.
fn get_mem_bound_reduce_num_groups_hint(device: &Device, kernel: &Kernel) -> Result<usize> {
  let DeviceInfoResult::MaxWorkGroupSize(max_lws) = device.info(DeviceInfo::MaxWorkGroupSize)?
  else {
    panic!("Wrong device info type!");
  };
  let KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(mult) =
    kernel.wg_info(*device, KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple)?
  else {
    panic!("Wrong kernel work group info type!");
  };

  // The result will always be a multiple of mult (assuming max_lws is...),
  // and if mult is a power of two, the result will also be a power of two.. assuming max_lws is.
  let hint = get_mem_bound_num_groups_hint(device)?;
  Ok(
    if mult.is_power_of_two() {
      hint.next_power_of_two()
    } else {
      hint.next_multiple_of(mult)
    }
    .clamp(mult, max_lws),
  )
}

// Computes the global work size to use in a 2-pass reduce operation.
// Returns a work size that results in scheduling a heuristically optimal number of groups.
// NOTE: Assumes a memory-bound typical grid-stride reduce, matching the above heuristics.
//       This function both splits ngroups across dimensions + converts to a gws.
fn get_mem_bound_reduce_gws_hint(
  device: &Device,
  kernel: &Kernel,
  lws: impl Into<SpatialDims>,
  reduce_size: impl Into<SpatialDims>,
) -> Result<SpatialDims> {
  // Returns a linear group hint less than budget, but always a power of 2.
  fn get_linear_ngroups_hint(lws: usize, size: usize, budget: usize) -> usize {
    (size / lws).prev_power_of_two().clamp(1, budget)
  }

  // Breaks budget across dimensions s.t. the product equals budget.
  // NOTE: For simplicity, we coerce budget to be a power of two here...
  let budget = get_mem_bound_reduce_num_groups_hint(device, kernel)?.prev_power_of_two();
  let Some((x, y, z)) = lws
    .into()
    .to_lens()?
    .into_iter()
    .zip(reduce_size.into().to_lens()?.into_iter())
    .scan(budget, |budget, (ld, sd)| {
      let hint = get_linear_ngroups_hint(ld, sd, *budget);
      *budget /= hint;
      Some(hint * ld) // The hinted gws = ngroups * lws
    })
    .collect_tuple()
  else {
    panic!("Expected exactly 3 usize when iterating SpatialDims")
  };
  Ok(SpatialDims::Three(x, y, z).reduce())
}

/*
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
*/

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
    .dims(1)
    .build()?;

  // Modeling the map as a 2D image.
  let map = Image::<i8>::builder()
    .queue(proque.queue().clone())
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
  let buffer = Image::<i8>::builder()
    .queue(proque.queue().clone())
    .image_type(MemObjectType::Image2d)
    .dims(map.dims())
    .channel_order(ImageChannelOrder::Luminance)
    .channel_data_type(ImageChannelDataType::SignedInt8)
    .build()?;

  // Hardcoding a reasonably efficient working size for now.
  let lws = SpatialDims::Two(32, 32);
  let find_accessible = proque
    .kernel_builder("find_accessible")
    .global_work_size(map.dims().to_padded_dims(lws)?)
    .local_work_size(lws)
    .arg(&map)
    .arg(4)
    .arg(&buffer)
    .build()?;
  let find_all_accessible = proque
    .kernel_builder("find_all_accessible")
    .global_work_size(1)
    .local_work_size(1)
    .arg(&map)
    .arg(&map)
    .arg(4)
    .arg(&buffer)
    .arg(&buffer)
    .build()?;

  let count_map = proque.buffer_builder::<i64>().fill_val(0).build()?;
  let count_buf = proque.buffer_builder::<i64>().fill_val(0).build()?;
  let mut count_accessible_map = proque
    .kernel_builder("count_accessible")
    .local_work_size(lws)
    .arg(&map)
    .arg(&count_map)
    .arg_named("scratch", None::<&Buffer<i64>>) // We don't know what size yet
    .build()?;
  let mut count_accessible_buf = proque
    .kernel_builder("count_accessible")
    .local_work_size(lws)
    .arg(&buffer)
    .arg(&count_buf)
    .arg_named("scratch", None::<&Buffer<i64>>) // We don't know what size yet
    .build()?;

  // Both images are identically sized + kernels will run sequentially (scartch can be shared).
  let gws =
    get_mem_bound_reduce_gws_hint(&proque.device(), &count_accessible_buf, lws, buffer.dims())?;
  count_accessible_map.set_default_global_work_size(gws);
  count_accessible_buf.set_default_global_work_size(gws);
  let scratch = proque
    .buffer_builder::<i64>()
    .len(gws.to_len() / lws.to_len())
    .fill_val(0)
    .build()?;
  count_accessible_map.set_arg("scratch", Some(&scratch))?;
  count_accessible_buf.set_arg("scratch", Some(&scratch))?;

  unsafe {
    match config.part {
      Part::One => {
        find_accessible.enq()?;
        count_accessible_buf.enq()?;
      }
      Part::Two => {
        find_all_accessible.enq()?;
        count_accessible_map.enq()?;
        count_accessible_buf.enq()?;
      }
    }
  }

  debug!("Map: {:?}", img2vec(&map)?);
  debug!("Buffer: {:?}", img2vec(&buffer)?);
  debug!("Scratch: {:?}", buf2vec(&scratch)?);

  // In part two, we don't know which image had the final result,
  // but we can just take the larger of the two counts.
  let cmap = buf2vec(&count_map)?.into_iter().exactly_one()?;
  let cbuf = buf2vec(&count_buf)?.into_iter().exactly_one()?;
  println!("Sum: {}", std::cmp::max(cmap, cbuf));

  Ok(())
}
