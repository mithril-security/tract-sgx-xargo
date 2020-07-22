extern crate ansi_term;
extern crate box_drawing;
extern crate clap;
#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate log;
extern crate atty;
extern crate env_logger;
extern crate pbr;
#[macro_use]
extern crate serde_derive;
extern crate tract_core;
#[cfg(feature = "onnx")]
extern crate tract_onnx;
#[cfg(feature = "tf")]
extern crate tract_tensorflow;

#[macro_use]
mod macros;

use std::process;
#[allow(unused_imports)]
use tract_itertools::Itertools;

use tract_core::internal::*;
use tract_hir::internal::*;

use crate::errors::*;
use crate::model::Model;

use readings_probe::*;

mod annotations;
mod bench;
mod compare;
mod cost;
mod display_params;
mod draw;
mod dump;
mod errors;
mod export;
mod model;
mod optimize_check;
mod params;
mod profile;
mod run;
mod stream_check;
mod tensor;
mod terminal;
mod utils;

use params::*;

readings_probe::instrumented_allocator!();

fn info_usage(stage: &str, probe: Option<&Probe>) {
    if let Some(mon) = probe {
        let _ = mon.log_event(stage);
    }
    if log::log_enabled!(log::Level::Info) {
        let usage = readings_probe::get_os_readings().unwrap();
        info!(
            "Resource usage {}: vsz:{} rsz:{} rszmax:{}",
            stage, usage.virtual_size, usage.resident_size, usage.resident_size_max
        );
    }
}

/// Entrypoint for the command-line interface.
fn main() {
    use clap::*;
    let mut app = clap_app!(("tract") =>
    (author: "Romain Liautaud <romain.liautaud@snips.ai>")
    (author: "Mathieu Poumeyrol <mathieu.poumeyrol@snips.ai>")
    (version: crate_version!())
    (about: "Tract command line interface")

    (@setting DeriveDisplayOrder)
    (@setting AllowLeadingHyphen)

    (@arg readings: --readings "Start readings instrumentation")
    (@arg readings_heartbeat: --("readings-heartbeat") +takes_value
     default_value("5") "Set heartbeat (ms)")

    (@arg model: +takes_value "Sets the model to use")

    (@arg format: -f +takes_value
     "Hint the model format ('kaldi', 'onnx' or 'tf') instead of guess from extension.")

    (@arg input: -i --input +takes_value +multiple number_of_values(1)
     "Set input shape and type (@file.pb or @file.npz:thing.npy or 3x4xi32).")

    (@arg const_input: --("const-input") +takes_value +multiple number_of_values(1)
     "Treat input as a Const (by name), retaining its value.")

    (@arg input_bundle: --("input-bundle") +takes_value +multiple number_of_values(1)
     "Path to an input container (.npz)")

    (@arg kaldi_adjust_final_offset: --("kaldi-adjust-final-offset") +takes_value
     "Adjust value of final offset in network (for reproducibility)")

    (@arg kaldi_downsample: --("kaldi-downsample") +takes_value
     "Add a subsampling to output on axis 0")

    (@arg kaldi_left_context: --("kaldi-left-context") +takes_value
     "Add lines of left context to input (dupping first time frame)")

    (@arg kaldi_right_context: --("kaldi-right-context") +takes_value
     "Add lines of right context to input (dupping last time frame)")

    (@arg onnx_test_data_set: --("onnx-test-data-set") +takes_value
     "Use onnx-test data-set as input (expect test_data_set_N dir with input_X.pb, etc. inside)")

    (@arg input_node: --("input-node") +takes_value +multiple number_of_values(1)
     "Override input nodes names (auto-detects otherwise).")

    (@arg tf_initializer_output_node: --("tf-initializer-output-node") +takes_value +multiple number_of_values(1)
     "Set an initializer node")

    (@arg output_node: --("output-node") +takes_value +multiple number_of_values(1)
     "Override output nodes name (auto-detects otherwise).")

    (@arg override_fact: --("override-fact") +takes_value +multiple number_of_values(1)
     "Override a fact.")

    (@arg analyse_fail_fast: --("analyse-fail-fast") "Stop analyse at first error.")
    (@arg recursive: --recursive "Apply to sub graphes")

    (@arg proto: --proto "Keep proto model around after parse")
    (@arg determinize: --determinize "Enforce a seed in random operator")

    (@arg partial: --partial "Before analyse, eliminate dead branches")

    (@arg pass: --pass +takes_value
     possible_values(&["load", "analyse", "incorporate", "type", "declutter",
                     "concretize-stream-dim", "concretize-stream-dim-declutter",
                     "pulse", "pulse-to-type", "pulse-declutter",
                     "optimize"])
     "Pass to stop preprocessing after.")

    (@arg optimize: -O --optimize "Optimize before running")
    (@arg pulse: --pulse +takes_value "Translate to pulse network")
    (@arg concretize_stream_dim: --("concretize-stream-dim") +takes_value "Replace streaming dim by a concrete value")

    (@arg verbosity: -v ... "Sets the level of verbosity.")

    (@arg machine_friendly: --("machine-friendly") "Machine friendly output")

    (@arg list_ops: --("list-ops") "List all known operators")
    );

    let compare = clap::SubCommand::with_name("compare")
        .long_about("Compares the output of tract and tensorflow on randomly generated input.")
        .arg(
            Arg::with_name("cumulative")
                .long("cumulative")
                .takes_value(false)
                .help("Do not reset with reference values at each node"),
        )
        .arg(
            Arg::with_name("resilient")
                .long("resilient")
                .takes_value(false)
                .help("Try nodes one per one to mitigate crashes"),
        );
    app = app.subcommand(output_options(compare));

    let compare_npz = clap::SubCommand::with_name("compare-npz")
        .long_about("Compares the output of tract to a refrence npz file.")
        .arg(
            Arg::with_name("cumulative")
                .long("cumulative")
                .takes_value(false)
                .help("Do not reset with reference values at each node"),
        )
        .arg(Arg::with_name("npz").takes_value(true).required(true).help("Npz filename"));
    app = app.subcommand(output_options(compare_npz));

    let compare_pbdir = clap::SubCommand::with_name("compare-pbdir")
        .long_about(
            "Compares the output of tract to a refrence directory of onnx protobufs tensors files.",
        )
        .arg(
            Arg::with_name("cumulative")
                .long("cumulative")
                .takes_value(false)
                .help("Do not reset with reference values at each node"),
        )
        .arg(Arg::with_name("pbdir").takes_value(true).required(true).help("protobuf dir"));
    app = app.subcommand(output_options(compare_pbdir));

    let bench = clap::SubCommand::with_name("bench")
        .long_about("Benchmarks tract on randomly generated input.");
    let bench = output_options(bench);
    let bench = benchlimits_options(bench);
    app = app.subcommand(bench);

    let criterion = clap::SubCommand::with_name("criterion")
        .long_about("Benchmarks tract on randomly generated input using criterion.");
    app = app.subcommand(criterion);

    let dump = clap::SubCommand::with_name("dump")
        .long_about("Dumps the Tensorflow graph in human readable form.")
        .arg(Arg::with_name("cost").long("cost").help("Include const information"))
        .arg(Arg::with_name("profile").long("profile").help("Include results for profile run"))
        .arg(
            Arg::with_name("assert-cost")
            .takes_value(true)
            .long("assert-cost")
            .help("Checks computed against the provided value (form: \"FMA(F32)=2060448 DIV(F32)=24576\")")
            )
        .arg(
            Arg::with_name("assert-output")
            .takes_value(true)
            .long("assert-output")
            .help("Fact to check the ouput tensor against (@filename, or 3x4xf32)"),
            )
        .arg(
            Arg::with_name("assert-output-fact")
            .takes_value(true)
            .long("assert-output-fact")
            .help("Infered shape and datum type must match exactly this"),
            )
        .arg(
            Arg::with_name("inner")
            .takes_value(true)
            .number_of_values(1)
            .multiple(true)
            .long("inner")
            .help("Navigate to a sub-model"),
            );
    let dump = output_options(dump);
    let dump = benchlimits_options(dump);
    app = app.subcommand(dump);

    let run = clap::SubCommand::with_name("run")
        .long_about("Run the graph")
        .arg(Arg::with_name("dump").long("dump").help("Show output"))
        .arg(Arg::with_name("steps").long("steps").help("Show all inputs and outputs"))
        .arg(
            Arg::with_name("assert-output-bundle")
                .takes_value(true)
                .long("assert-output-bundle")
                .help("Checks values against these tensor (.npz)"),
        )
        .arg(
            Arg::with_name("assert-output")
                .takes_value(true)
                .long("assert-output")
                .help("Fact to check the ouput tensor against (@filename, or 3x4xf32)"),
        )
        .arg(
            Arg::with_name("assert-output-fact")
                .takes_value(true)
                .long("assert-output-fact")
                .help("Infered shape and datum type must match exactly this"),
        );
    app = app.subcommand(output_options(run));

    let optimize = clap::SubCommand::with_name("optimize").help("Optimize the graph");
    app = app.subcommand(output_options(optimize));

    let optimize_check = clap::SubCommand::with_name("optimize-check")
        .long_about("Compare output of optimized and un-optimized graph");
    app = app.subcommand(output_options(optimize_check));

    let stream_check = clap::SubCommand::with_name("stream-check")
        .long_about("Compare output of streamed and regular exec");
    app = app.subcommand(output_options(stream_check));

    let matches = app.get_matches();

    let probe = if matches.is_present("readings") {
        let file = std::fs::File::create("readings.out").unwrap();
        let mut probe = Probe::new(file).unwrap();
        probe.register_i64("progress").unwrap();
        let heartbeat = matches.value_of("readings_heartbeat").unwrap().parse::<f32>().unwrap();
        probe.spawn_heartbeat(std::time::Duration::from_secs_f32(heartbeat / 1000.0)).unwrap();
        Some(probe)
    } else {
        None
    };

    if ::std::env::var("RUST_LOG").is_err() {
        let level = match matches.occurrences_of("verbosity") {
            0 => "cli=warn,tract=warn",
            1 => "cli=info,tract=info",
            2 => "cli=debug,tract=debug",
            _ => "cli=trace,tract=trace",
        };
        ::std::env::set_var("RUST_LOG", level);
    }

    let env = env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "warn");

    env_logger::Builder::from_env(env).format_timestamp_nanos().init();
    info_usage("init", probe.as_ref());

    if let Err(e) = handle(matches, probe.as_ref()) {
        use error_chain::ChainedError;
        error!("{}", e.display_chain());
        process::exit(1)
    }

    info_usage("done", probe.as_ref());
}

fn benchlimits_options<'a, 'b>(command: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
    use clap::*;
    command
        .arg(
            Arg::with_name("max_iters")
                .takes_value(true)
                .long("max-iters")
                .short("n")
                .help("Sets the maximum number of iterations for each node [default: 100_000]."),
        )
        .arg(
            Arg::with_name("max-time")
                .takes_value(true)
                .long("max-time")
                .help("Sets the maximum execution time for each node (in ms) [default: 5000]."),
        )
}

fn output_options<'a, 'b>(command: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
    use clap::*;
    command
        .arg(
            Arg::with_name("natural-order")
                .long("natural-order")
                .help("dump nodes in id order instead of evaluation order"),
        )
        .arg(Arg::with_name("quiet").short("q").long("quiet").help("don't dump"))
        .arg(Arg::with_name("debug-op").long("debug-op").help("show debug dump for each op"))
        .arg(
            Arg::with_name("node_id")
                .long("node-id")
                .takes_value(true)
                .help("Select a node to dump"),
        )
        .arg(
            Arg::with_name("successors")
                .long("successors")
                .takes_value(true)
                .help("Show successors of node"),
        )
        .arg(
            Arg::with_name("op_name")
                .long("op-name")
                .takes_value(true)
                .help("Select one op to dump"),
        )
        .arg(
            Arg::with_name("node_name")
                .long("node-name")
                .takes_value(true)
                .help("Select one node to dump"),
        )
        .arg(Arg::with_name("const").long("const").help("also display consts nodes"))
        .arg(Arg::with_name("info").long("info").help("show op inner information"))
        .arg(Arg::with_name("io-long").long("io-long").help("show full i/o information"))
        .arg(Arg::with_name("io-none").long("io-none").help("hide i/o information"))
        .arg(Arg::with_name("json").long("json").help("dump performance info as json"))
        .arg(Arg::with_name("outlet-labels").long("outlet-labels").help("display outlet labels"))
        .arg(
            Arg::with_name("invariants")
                .takes_value(false)
                .long("invariants")
                .help("Display operators invariants"),
        )
}

/// Handles the command-line input.
fn handle(matches: clap::ArgMatches, probe: Option<&Probe>) -> CliResult<()> {
    if matches.is_present("list_ops") {
        #[cfg(feature = "onnx")]
        {
            let onnx = tract_onnx::onnx();
            let names = onnx.op_register.0.keys().sorted().into_iter().join(", ");
            println!("Onnx:\n");
            println!("{}", names);
            println!("\n");
        }
        #[cfg(feature = "tf")]
        {
            let tf = tract_tensorflow::tensorflow();
            let names = tf.op_register.0.keys().sorted().into_iter().join(", ");
            println!("Tensorflow:\n");
            println!("{}", names);
            println!("\n");
        }
        return Ok(());
    }

    #[allow(unused_mut)]
    let mut params = Parameters::from_clap(&matches, probe)?;

    let mut need_optimisations = false;

    match matches.subcommand() {
        ("bench", Some(m)) => {
            need_optimisations = true;
            bench::handle(&params, &BenchLimits::from_clap(&m)?, probe)
        }

        ("criterion", _) => {
            need_optimisations = true;
            bench::criterion(&params)
        }

        #[cfg(feature = "conform")]
        ("compare", Some(m)) => compare::handle_tensorflow(
            m.is_present("cumulative"),
            m.is_present("resilient"),
            &mut params,
            display_params_from_clap(&matches, m)?,
        ),
        #[cfg(not(feature = "conform"))]
        ("compare", _) => bail!("Need conform feature to be able to run comparison"),

        ("compare-npz", Some(m)) => compare::handle_npz(
            m.is_present("cumulative"),
            m.value_of("npz").unwrap(),
            &params,
            display_params_from_clap(&matches, m)?,
        ),

        #[cfg(feature = "onnx")]
        ("compare-pbdir", Some(m)) => compare::handle_pbdir(
            m.is_present("cumulative"),
            m.value_of("pbdir").unwrap(),
            &params,
            display_params_from_clap(&matches, m)?,
        ),

        ("run", Some(m)) => run::handle(&params, m.is_present("dump"), m.is_present("steps")),

        ("optimize-check", Some(m)) => {
            optimize_check::handle(&params, display_params_from_clap(&matches, m)?)
        }

        ("stream-check", Some(m)) => {
            stream_check::handle(&params, &display_params_from_clap(&matches, m)?)
        }

        ("", None) => dump::handle(
            &params,
            &display_params_from_clap(&matches, &clap::ArgMatches::default())?,
            &clap::ArgMatches::default(),
            &BenchLimits::from_clap(&clap::ArgMatches::default())?,
            vec![],
        ),

        ("dump", Some(m)) => {
            need_optimisations = m.is_present("profile");
            let inner = m
                .values_of("inner")
                .map(|ss| ss.map(|s| s.to_string()).collect())
                .unwrap_or(vec![]);
            dump::handle(
                &params,
                &display_params_from_clap(&matches, m)?,
                m,
                &BenchLimits::from_clap(&m)?,
                inner,
            )
        }

        (s, _) => bail!("Unknown subcommand {}.", s),
    }?;

    if need_optimisations {
        let style = ansi_term::Style::new().fg(ansi_term::Color::Red).bold();
        if !matches.is_present("optimize") {
            warn!("{}", style.paint("Profiling an un-optimized network. Consider adding -O."));
        }
        if cfg!(debug_assertions) {
            warn!("{}", style.paint("Profiling a debug build of tract!"));
        }
    }

    if let Some(e) = params.analyse_error {
        Err(e)?
    }
    Ok(())
}
