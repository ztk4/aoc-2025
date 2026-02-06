//! AoC Day 4
#![feature(result_flattening, macro_metavar_expr)]
use clap::{Args, arg};
use color_eyre::eyre::{Result, eyre};
use libaoc::*;
use log::*;
use num_traits::FromPrimitive;
use ocl::{core::ClDeviceIdPtr, flags::*, prm::*, *};
use std::fs;

/// Allows queue creation via clCreateCommandQueueWithProperties.
/// This method allows for additional property values compared to clCreateCommandQueue.
/// TODO: Make this more general (to support other queue properties).
/// NOTE: I probably will not expand this since device-side queues are not officially supported on
///       my hardware.
fn create_command_queue_with_properties(
  context: &Context,
  device: Device,
  cq_properties: Option<CommandQueueProperties>,
) -> Result<core::CommandQueue> {
  // NOTE: This is essentially the impl of Queue::new, with the nested calls to
  // core::create_command_queue and private utilities inlined -> calling the correct ffi method.
  core::verify_context(context)?;
  // This list of cl_queue_properties is a NULL-terminated dictionary of pairs.
  // Every even index is a property constant, and every odd index is its value.
  let props = [
    // QUEUE_PROPERTIES
    ffi::CL_QUEUE_PROPERTIES as ffi::cl_queue_properties,
    cq_properties.map_or(0, |p| p.bits()) as ffi::cl_queue_properties,
    // NULL TERMINATOR
    0 as ffi::cl_queue_properties,
  ];
  debug!("Queue props: {:?}", props);

  let mut err: ffi::cl_int = 0;
  let queue = unsafe {
    ffi::clCreateCommandQueueWithProperties(
      context.as_ptr(),
      device.as_ptr(),
      props.as_ptr(),
      &mut err,
    )
  };

  if core::Status::CL_SUCCESS as i32 != err {
    return Err(eyre!(
      "clCreateCommandQueueWithProperties failed with {}",
      core::Status::from_i32(err).map_or(format!("<invalid code {err}>"), |s| format!("{:?}", s))
    ));
  }

  Ok(unsafe { core::CommandQueue::from_raw_create_ptr(queue) })
}

#[derive(Args)]
struct Options {
  /// The maximum number of times a kenel enqueued by the host may enqueue kernels on device.
  /// NOTE: For my RTX 3080, on-device enqueue is not officially supported, but seems to tolerate
  /// up to ~24 enqueues.
  #[arg(default_value_t = 20)]
  max_ondevice_enqueues: i32,
}

fn main() -> Result<()> {
  env_logger::init();
  color_eyre::install()?;

  let config = create_config!(challenge_args: Options)?;
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
  // We're using on-device enqueue, so we need to create a device-side queue for that.
  let _device_queue = create_command_queue_with_properties(
    &proque.context(),
    proque.device(),
    Some(
      CommandQueueProperties::ON_DEVICE_DEFAULT  // Will be returned by get_default_queue()
        | CommandQueueProperties::ON_DEVICE      // Reqd for on-device queues
        | CommandQueueProperties::OUT_OF_ORDER_EXEC_MODE_ENABLE, // Reqd for on-device queues
    ),
  )?;

  let i2_dims = Int2::new(input[0].len() as i32, input.len() as i32);
  let dims: SpatialDims = i2_dims.first_chunk::<2>().unwrap().into();
  // Modeling the map as a buffer.
  let map = proque
    .buffer_builder::<i8>()
    .len(dims.to_len())
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
  let buffer = proque
    .buffer_builder::<i8>()
    .len(dims.to_len())
    .fill_val(0)
    .build()?;
  let finished = proque.buffer_builder::<i32>().fill_val(0).build()?;

  // Hardcoding a reasonably efficient working size for now.
  let lws = SpatialDims::Two(32, 32);
  let find_accessible = proque
    .kernel_builder("find_accessible")
    .global_work_size(dims.to_padded_dims(lws)?)
    .local_work_size(lws)
    .arg(&map)
    .arg(i2_dims)
    .arg(4)
    .arg(&buffer)
    .build()?;
  let find_all_accessible = proque
    .kernel_builder("find_all_accessible")
    .global_work_size(1)
    .local_work_size(1)
    .arg(&map)
    .arg(i2_dims)
    .arg(4)
    .arg(&buffer)
    // Each pass of this kernel schedules 2 kernels on-device.
    // But we also want this to be an odd number s.t. the unfinished result is always in map.
    .arg(match config.challenge_args.max_ondevice_enqueues / 2 {
      n if n % 2 == 0 => n - 1,
      n => n,
    })
    .arg(&finished)
    .build()?;

  let count_map = proque.buffer_builder::<i64>().fill_val(0).build()?;
  let count_buf = proque.buffer_builder::<i64>().fill_val(0).build()?;
  let mut count_accessible_map = proque
    .kernel_builder("count_accessible")
    .local_work_size(lws)
    .arg(&map)
    .arg(i2_dims)
    .arg(&count_map)
    .arg_named("scratch", None::<&Buffer<i64>>) // We don't know what size yet
    .build()?;
  let mut count_accessible_buf = proque
    .kernel_builder("count_accessible")
    .local_work_size(lws)
    .arg(&buffer)
    .arg(i2_dims)
    .arg(&count_buf)
    .arg_named("scratch", None::<&Buffer<i64>>) // We don't know what size yet
    .build()?;

  // Both buffers are identically sized + kernels will run sequentially (scratch can be shared).
  let gws = get_mem_bound_reduce_gws_hint(&proque.device(), &count_accessible_buf, lws, dims)?;
  count_accessible_map.set_default_global_work_size(gws);
  count_accessible_buf.set_default_global_work_size(gws);
  let scratch = proque
    .buffer_builder::<i64>()
    .len(gws.to_len() / lws.to_len())
    .fill_val(0)
    .build()?;
  count_accessible_map.set_arg("scratch", Some(&scratch))?;
  count_accessible_buf.set_arg("scratch", Some(&scratch))?;

  let count_accessible2_map = proque
    .kernel_builder("count_accessible2")
    .global_work_size(scratch.len())
    .local_work_size(scratch.len())
    .arg(&scratch)
    .arg(&count_map)
    .build()?;
  let count_accessible2_buf = proque
    .kernel_builder("count_accessible2")
    .global_work_size(scratch.len())
    .local_work_size(scratch.len())
    .arg(&scratch)
    .arg(&count_buf)
    .build()?;

  match config.part {
    Part::One => unsafe {
      find_accessible.enq()?;
      count_accessible_buf.enq()?;
      count_accessible2_buf.enq()?;
    },
    Part::Two => {
      loop {
        unsafe {
          find_all_accessible.enq()?;
        }
        if buf2vec(&finished)?.into_iter().exactly_one()? > 0 {
          break;
        }
      }
      unsafe {
        count_accessible_map.enq()?;
        count_accessible2_map.enq()?;
        count_accessible_buf.enq()?;
        count_accessible2_buf.enq()?;
      }
    }
  }

  debug!("Map: {:?}", buf2vec(&map)?);
  debug!("Buffer: {:?}", buf2vec(&buffer)?);
  debug!("Scratch: {:?}", buf2vec(&scratch)?);

  // In part two, we don't know which image had the final result,
  // but we can just take the larger of the two counts.
  let cmap = buf2vec(&count_map)?.into_iter().exactly_one()?;
  let cbuf = buf2vec(&count_buf)?.into_iter().exactly_one()?;
  println!("Sum: {}", std::cmp::max(cmap, cbuf));

  Ok(())
}
