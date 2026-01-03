#![feature(
  file_buffered,
  macro_metavar_expr,
  result_flattening,
  pattern,
  try_trait_v2,
  try_trait_v2_residual
)]
use clap::{Parser, ValueEnum};
use color_eyre::eyre::{Result, eyre};
use futures::future::FutureExt; // How do you live without Future::then/map?
use ocl::{Buffer, OclPrm};
use smol::{LocalExecutor, channel, future};

use std::any::type_name;
use std::ops::{ControlFlow, FromResidual, Residual, Try};
use std::slice::Iter;
use std::str::{FromStr, Split, SplitWhitespace, pattern::Pattern};
use std::string::String;

// These imports are necessary to use most of the below utilities.
pub use color_eyre::eyre::WrapErr;
pub use itertools::Itertools;
pub use std::io::BufRead;

/// Configuration utilities based on CLI + environment.

/// Constructs a populated Config object for an example crate.
// NOTE: Using a macro to insert the ENV lookup macro into the calling crate.
#[macro_export]
macro_rules! create_config {
  (
    $(challenge_args: $type:ty)?  // Optional specification of additional args.
  ) => {
    <Config$(<$type>)?>::create(std::env!("CARGO_CRATE_NAME"))
  };
}

/// Configuration for a specific AoC Challenge.
pub struct Config<T: clap::Args = EmptyChallengeArgs> {
  /// Which day of AoC.
  pub day: u8,
  /// Which part of the challenge.
  pub part: Part,
  /// The file to read input from (buffered as a best-practice).
  pub input: std::io::BufReader<std::fs::File>,
  /// The local directory for this day's challenge.
  pub local_dir: std::path::PathBuf,
  /// Hints local work size for OpenCL.
  /// NOTE: This parameter may be ignored.
  pub group_size: usize,
  /// Custom challenge-specific arguments (if specified).
  pub challenge_args: T,
}

impl<T: clap::Args> Config<T> {
  pub fn create(example_crate: &str) -> Result<Self> {
    let args = Args::<T>::parse();
    let local_dir: std::path::PathBuf =
      [std::env!("CARGO_MANIFEST_DIR"), "examples", example_crate]
        .iter()
        .collect();

    // Assuming the challenge is implemented as an example crate named as a day number.
    // E.g. examples/12/main.rs for day 12.
    Ok(Config {
      day: example_crate
        .parse::<u8>()
        .wrap_err_with(|| format!("Crate name '{}' must be a day number.", example_crate))?,
      part: args.part,
      input: {
        let path = local_dir.join(args.input);
        std::fs::File::open_buffered(path.as_path())
          .wrap_err_with(|| format!("Bad path: {}", path.display()))?
      },
      local_dir,
      group_size: args.group_size,
      challenge_args: args.challenge_args,
    })
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Part {
  One,
  Two,
}

#[derive(clap::Args)]
pub struct EmptyChallengeArgs;

/// Generic args for an AoC Chall.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args<T: clap::Args = EmptyChallengeArgs> {
  /// Which part of the challenge to solve.
  #[arg(value_enum, short, long, default_value_t = Part::One)]
  part: Part,

  /// Input file, relative to the examples/ subdirectory for this day.
  #[arg(short, long)]
  input: String,

  /// A hint for what local work size to use for OpenCL.
  /// NOTE: This may be ignored.
  #[arg(short, long, default_value_t = 1024)]
  group_size: usize,

  /// Additional arguments supported by the specific AoC Chall.
  #[command(flatten)]
  challenge_args: T,
}

/// File parsing utilities that are generic across many challenges.

/// Parses the given tokens into a tuple.
/// The tuple given after `as` describes the result type.
/// If the number of items does not match, or if a parse fails, an error is returned.
#[macro_export]
macro_rules! tuple_parse_tokens {
  (
    $tokens:expr
    => ($($type:ty),+)
  ) => {
    // NOTE: Using an inline closure to catch `?` error propagation.
    (|| -> color_eyre::eyre::Result<($($type),+)> {
      // Unpack the line into a tuple.
      let tuple = $tokens
        // Supports something "like" `impl IntoIterator<Item: AsRef<str>>`
        .into_iter()
        .map(|tok| -> &str { tok.as_ref() })
        .collect_tuple::<($(${ignore($type)} _),+)>()
        .ok_or_else(||
          color_eyre::eyre::eyre!("Expected to parse exactly {} items", ${count($type)}))?;
      // Convert each tuple member via the corresponding parse call.
      Ok(($(
          tuple
            .${index()}
            .parse::<$type>()
            .wrap_err_with(|| {
              color_eyre::eyre::eyre!("Expected token `{}` to be parseable as a(n) {}",
                tuple.${index()},
                std::any::type_name::<$type>())
            })?
      ),+))
    })()
  };
}

// Honestly with how I'm using this, it's kind of silly.
pub enum DelimitedTokens<'src, P: Pattern> {
  FromSplit(Split<'src, P>),
  FromSplitWhitespace(SplitWhitespace<'src>),
  FromBytes(Iter<'src, u8>),
}
impl<'src, P: Pattern> DelimitedTokens<'src, P> {
  /// Makes tokens by splitting on a separator.
  pub fn by_sep(s: impl Into<&'src str>, sep: P) -> Self {
    Self::FromSplit(s.into().split(sep))
  }
}
impl<'src> DelimitedTokens<'src, [char; 0]> {
  /// Makes tokens by splitting on unicode whitespace.
  pub fn by_whitespace(s: impl Into<&'src str>) -> Self {
    Self::FromSplitWhitespace(s.into().split_whitespace())
  }
  /// Makes tokens from each individual ASCII character.
  /// NOTE: A non-ASCII string will cause this to crash.
  pub fn each_char(s: impl Into<&'src str>) -> Self {
    Self::FromBytes(s.into().as_bytes().iter())
  }
}
impl<'src, P: Pattern> Iterator for DelimitedTokens<'src, P> {
  type Item = &'src str;

  fn next(&mut self) -> Option<Self::Item> {
    match self {
      DelimitedTokens::FromSplit(ref mut it) => it.next(),
      DelimitedTokens::FromSplitWhitespace(ref mut it) => it.next(),
      DelimitedTokens::FromBytes(ref mut it) => it.next().map(|b| {
        std::str::from_utf8(std::slice::from_ref(b))
          .expect("DelimitedTokens::FromBytes requires each byte to be valid ASCII")
      }),
    }
  }
}
/// Parses the given tokens into a vector.
/// All items are parsed as the passed generic type.
/// If any parse fails, the first error is returned.
pub fn vec_parse_tokens<'a, T: FromStr>(tokens: impl IntoIterator<Item = &'a str>) -> Result<Vec<T>>
where
  Result<T, T::Err>: WrapErr<T, T::Err>,
{
  tokens
    .into_iter()
    .map(|tok| (tok, str::parse::<T>(tok)))
    .map(|(tok, res)| {
      res.wrap_err_with(|| eyre!("Failed to parse token `{tok}` as a(n) {}", type_name::<T>()))
    })
    .collect()
}

/// Parses all lines using the given parser.
pub fn parse_lines<P, R, T, E>(
  lines: impl IntoIterator<Item: AsRef<str>>,
  mut parser: P,
) -> impl Iterator<Item = Result<T>>
where
  P: FnMut(&'_ str) -> R,
  R: WrapErr<T, E>,
{
  lines.into_iter().map(move |line| {
    parser(line.as_ref()).wrap_err_with(|| eyre!("Failed to parse line `{}`", line.as_ref()))
  })
}

/// Same as parse_lines, but handling results.
pub fn parse_result_lines<'out, I, EI, P, R, T, ET>(
  rlines: impl IntoIterator<Item = Result<I, EI>, IntoIter: 'out>,
  parser: P,
) -> impl Iterator<Item = Result<T>> + 'out
where
  I: AsRef<str>,                 // Can be used as Item to call parse_lines
  Result<I, EI>: WrapErr<I, EI>, // Can be coerced to a Report.
  <Result<I, EI> as Try>::Residual: Residual<Result<T>>, // Can re-package EI with Result<T>.
  P: FnMut(&'_ str) -> R + 'out,
  R: WrapErr<T, ET> + 'out,
  T: 'out,
  ET: 'out,
{
  process_outputs_as_iterator(
    rlines
      .into_iter()
      .map(|rline| rline.wrap_err("Incoming error prevented parsing")),
    move |lines| parse_lines(lines, parser),
  )
  .map(|res| res.flatten()) // Flatten incoming errors w/ parser errors.
}

/// "Lift" a routine `f` taking an iterator of `Output`s and producing an arbitrary iterator
/// s.t. the routine can be applied to an iterator of `Try`s.
/// The `Output` cases are processed via the routine, while the `Residual` cases pass along.
pub fn process_outputs_as_iterator<'out, I, T, F, S>(
  iterable: I,
  f: F,
) -> ProcessOutputsAsIterator<'out, Option<S::Item>, T::Residual>
where
  I: IntoIterator<Item = T>,
  T: Try,
  F: FnOnce(TryIteratorWithResidualSideChannel<I::IntoIter, T::Residual>) -> S,
  S: IntoIterator + 'out,
  ProcessOutputsAsIterator<'out, Option<S::Item>, T::Residual>: Iterator, // output will be iterable.
{
  // Thread-local executor + sync primitives.
  let (output_send, output_recv) = channel::bounded(1);
  let (residual_send, residual_recv) = channel::bounded(1);
  let local = LocalExecutor::new();
  // An adapter for I which only emits the Output values.
  // Residuals are side-channeled through residual_send.
  let input_fork = TryIteratorWithResidualSideChannel {
    source: iterable.into_iter(),
    side_channel: residual_send,
  };
  // Schedule an ongoing task to attempt running the result of F indefinitely.
  // Directly returns outputs via output_send, while the nested iterator returns residuals.
  // NOTE: Tasks on a thread-local executor will NOT run until the set run.
  let mut it = f(input_fork).into_iter();
  local
    .spawn(async move {
      loop {
        output_send.send(it.next()).await.unwrap(); // Expected to never panic.
      }
    })
    .detach();
  // The final adapter which consumes the above channels, merging the results.
  ProcessOutputsAsIterator {
    output_channel: output_recv,
    residual_channel: residual_recv,
    local,
  }
}

/// Iterator for implementing the above adapter.
/// Selects values from either channel, and returns it composed as a Try.
pub struct ProcessOutputsAsIterator<'local, O, R> {
  output_channel: channel::Receiver<O>,
  residual_channel: channel::Receiver<R>,
  local: LocalExecutor<'local>,
}
impl<'local, O, R> Iterator for ProcessOutputsAsIterator<'local, Option<O>, R>
where
  R: Residual<O>,
{
  type Item = <R as Residual<O>>::TryType;

  fn next(&mut self) -> Option<Self::Item> {
    future::block_on(self.local.run(async {
      future::or(
        self
          .output_channel
          .recv()
          .map(|output| output.unwrap().map(Self::Item::from_output)),
        self
          .residual_channel
          .recv()
          .map(|residual| Some(Self::Item::from_residual(residual.unwrap()))),
      )
      .await
    }))
  }
}

/// Co-iterator with `ProcessOutputsAsIterator` which forwards Outputs,
/// while passing Residuals through the side channel.
pub struct TryIteratorWithResidualSideChannel<I, R> {
  source: I,
  side_channel: channel::Sender<R>,
}
impl<I, T> Iterator for TryIteratorWithResidualSideChannel<I, T::Residual>
where
  I: Iterator<Item = T>,
  T: Try,
{
  type Item = T::Output;

  fn next(&mut self) -> Option<Self::Item> {
    // While the source produces values...
    while let Some(t) = self.source.next() {
      match t.branch() {
        // ... either forward residuals to the side channel (expect to never panic)...
        ControlFlow::Break(residual) => self.side_channel.send_blocking(residual).unwrap(),
        // ... or return outputs.
        ControlFlow::Continue(output) => return Some(output),
      };
    }

    None // source is finished.
  }
}

/// Generic helpers useful for multiple challenges.

/// Allows an operation on a copy of a Cell's interior by reference.
/// Note that the cell's interior remains well defined (T::default()) during f/panic.
/// Any mutations via apply_mut are visible in the Cell after apply_mut returns.
/// Expectation: under most conditions, the take/replace is optimized out.
pub trait CellApplyExt<T> {
  fn apply<R>(&self, f: impl FnOnce(&T) -> R) -> R;
  fn apply_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> R;
}
impl<T: Default> CellApplyExt<T> for std::cell::Cell<T> {
  #[inline]
  fn apply<R>(&self, f: impl FnOnce(&T) -> R) -> R {
    let t = self.take();
    let r = f(&t);
    self.set(t);
    r
  }
  #[inline]
  fn apply_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
    let mut t = self.take();
    let r = f(&mut t);
    self.set(t);
    r
  }
}

/// Helpers for working with ocl.

/// Simple conversion of buffer to vector.
/// NOTE: Performs a memory transfer from the GPU -> requires buf have a default queue.
pub fn buf2vec<T: OclPrm>(buf: &Buffer<T>) -> Result<Vec<T>> {
  let mut vec = vec![T::default(); buf.len()];
  buf.read(&mut vec).enq()?;
  Ok(vec)
}

/// Some minimal unit testing.

#[cfg(test)]
mod tests {
  use super::*;
  use color_eyre::eyre::Ok;

  #[test]
  fn verify_args() {
    use clap::CommandFactory;
    <Args>::command().debug_assert();
  }

  #[test]
  fn verify_extended_args() {
    use clap::CommandFactory;
    #[derive(clap::Args)]
    struct MyArgs {
      #[arg(short)]
      test: u64,
    }

    Args::<MyArgs>::command().debug_assert();
  }

  #[test]
  fn by_whitespace_splits_by_whitespace() {
    assert_eq!(
      DelimitedTokens::by_whitespace("1  2\n3\r\n4\t5")
        .into_iter()
        .collect::<Vec<_>>(),
      vec!["1", "2", "3", "4", "5"]
    )
  }

  #[test]
  fn by_sep_splits_by_separator() {
    assert_eq!(
      DelimitedTokens::by_sep("1&&2&&3&&4", "&&")
        .into_iter()
        .collect::<Vec<_>>(),
      vec!["1", "2", "3", "4"]
    )
  }

  #[test]
  fn tuple_parse_line_handles_mixed_types() -> Result<()> {
    assert_eq!(
      tuple_parse_tokens!(DelimitedTokens::by_whitespace("1 2 3") => (i64, f32, char))?,
      (1i64, 2f32, '3')
    );
    Ok(())
  }

  #[test]
  fn tuple_parse_line_errors_for_wrong_number_of_elements() -> Result<()> {
    assert!(tuple_parse_tokens!(DelimitedTokens::by_whitespace("1 2 3") => (i64, f32)).is_err());
    assert!(
      tuple_parse_tokens!(DelimitedTokens::by_whitespace("1 2 3") => (i64, f32, char, i64))
        .is_err()
    );
    Ok(())
  }

  #[test]
  fn tuple_parse_line_errors_when_parse_fails() -> Result<()> {
    assert!(
      tuple_parse_tokens!(DelimitedTokens::by_whitespace("a b c") => (char, i64, char)).is_err()
    );
    Ok(())
  }

  #[test]
  fn vec_parse_line_handles_normal_tokens() -> Result<()> {
    assert_eq!(
      vec_parse_tokens::<i64>(DelimitedTokens::by_whitespace("1 2  3 4  5"))?,
      vec![1, 2, 3, 4, 5]
    );
    Ok(())
  }

  #[test]
  fn vec_parse_line_errors_when_parse_fails() -> Result<()> {
    assert!(vec_parse_tokens::<i64>(DelimitedTokens::by_whitespace("1 2 test 4 5")).is_err());
    Ok(())
  }

  #[test]
  fn parse_lines_handles_tuples() -> Result<()> {
    assert_eq!(
      parse_lines(
        "a 2 -3\nd 5 -6\ng 8 -9".lines(),
        |line| tuple_parse_tokens!(DelimitedTokens::by_whitespace(line) => (char, u64, i64))
      )
      .collect::<Result<Vec<_>>>()?,
      vec![('a', 2u64, -3i64), ('d', 5u64, -6i64), ('g', 8u64, -9i64)]
    );
    Ok(())
  }

  #[test]
  fn parse_lines_errors_for_tuple_parse_error() -> Result<()> {
    assert!(
      parse_lines("1 2 3\n4 5\n6 7 8".lines(), |line| tuple_parse_tokens!(
        DelimitedTokens::by_whitespace(line) => (u64, u64, u64)
      ))
      .collect::<Result<Vec<_>>>()
      .is_err()
    );
    Ok(())
  }

  #[test]
  fn parse_lines_handles_vecs() -> Result<()> {
    assert_eq!(
      parse_lines(
        "1 2 3\n4 5 6 7\n8 9".lines(),
        |line| vec_parse_tokens::<i64>(DelimitedTokens::by_whitespace(line))
      )
      .collect::<Result<Vec<_>>>()?,
      vec![vec![1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]]
    );
    Ok(())
  }

  #[test]
  fn parse_lines_errors_for_vec_parse_error() -> Result<()> {
    assert!(
      parse_lines("1 2 3\n-4 5 6".lines(), |line| vec_parse_tokens::<u64>(
        DelimitedTokens::by_whitespace(line)
      ))
      .collect::<Result<Vec<_>>>()
      .is_err()
    );
    Ok(())
  }

  #[test]
  fn parse_result_lines_handles_tuples() -> Result<()> {
    assert_eq!(
      parse_result_lines(
        [Ok("a 1"), Ok("b 2"), Ok("c 3")],
        |line| tuple_parse_tokens!(DelimitedTokens::by_whitespace(line) => (char, u64))
      )
      .collect::<Result<Vec<_>>>()?,
      [('a', 1), ('b', 2), ('c', 3)]
    );
    Ok(())
  }

  #[test]
  fn parse_result_lines_handles_vecs() -> Result<()> {
    assert_eq!(
      parse_result_lines([Ok("1 2 3"), Ok("4 5 6"), Ok("7 8 9")], |line| {
        vec_parse_tokens::<i64>(DelimitedTokens::by_whitespace(line))
      })
      .collect::<Result<Vec<_>>>()?,
      [vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
    );
    Ok(())
  }

  #[test]
  fn parse_result_lines_returns_input_error() -> Result<()> {
    assert!(
      parse_result_lines(
        [Ok("a 1"), Err(eyre!("failed")), Ok("c 3")],
        |line| tuple_parse_tokens!(DelimitedTokens::by_whitespace(line) => (char, u64))
      )
      .collect::<Result<Vec<_>>>()
      .is_err()
    );
    Ok(())
  }

  #[test]
  fn parse_result_lines_returns_parse_error() -> Result<()> {
    assert!(
      parse_result_lines(
        [Ok("a 1"), Ok("BAD FORMAT"), Ok("c 3")],
        |line| tuple_parse_tokens!(DelimitedTokens::by_whitespace(line) => (char, u64))
      )
      .collect::<Result<Vec<_>>>()
      .is_err()
    );
    Ok(())
  }
}
