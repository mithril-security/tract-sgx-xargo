use tract_onnx::prelude::*;
use env_logger::Env;

fn main() -> TractResult<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("trace")).init();
    let model = tract_onnx::onnx()
        // load the model
        .model_for_path("/home/mithril-dev/gpt/gpt-neo-1.3B.onnx")?
        // specify input type and shape
        .with_input_fact(0, i64::fact(&vec![1, 3]).into())?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;


    println!("model loading passed");
    let mut hello_world_tensor = tract_ndarray::Array2::<i64>::zeros((1,3));
    let input : Tensor = hello_world_tensor.into();

    // run the model on the input
    let result = model.run(tvec!(input))?;
    println!("inference passed");
    Ok(())
}
