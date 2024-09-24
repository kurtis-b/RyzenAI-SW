@echo off
setlocal

:: Create the benchmark_runs directory if it doesn't exist
if not exist "benchmark_runs" (
    mkdir "benchmark_runs"
)

:: Number of times to run the Python script
set /a num_runs=10

:: Define the prefix of the output file 
set "prefixcpu=onnxruntime_profile_cpu_run_"
set "prefixnpu=onnxruntime_profile_npu_run_"

:: Loop to run Llama 2 inference on the CPU
for /L %%i in (1,1,%num_runs%) do (
    echo Running benchmark %%i...

    :: Run the Python script 
    python .\infer.py --model_name meta-llama/Llama-2-7b-hf --target cpu --model_dir llama2/quant --task benchmark --ort_trace --profile_file_prefix %prefixcpu%%%i

    :: Find the file with the specified prefix
    for %%f in ("%prefixcpu%%%i*") do (
        echo Found file: %%f
        :: Copy the file to the benchmark_runs directory with a unique name
        copy "%%f" "benchmark_runs\%prefixcpu%%%i%%~xf"
        del %%f
    )
)

:: Loop to run Llama 2 inference on the NPU
for /L %%i in (1,1,%num_runs%) do (
    echo Running benchmark %%i...

    :: Run the Python script 
    python .\infer.py --model_name meta-llama/Llama-2-7b-hf --target cpu --model_dir llama2/quant --task benchmark --ort_trace --profile_file_prefix %prefixnpu%%%i

    :: Find the file with the specified prefix
    for %%f in ("%prefixnpu%%%i*") do (
        echo Found file: %%f
        :: Copy the file to the benchmark_runs directory with a unique name
        copy "%%f" "benchmark_runs\%prefixnpu%%%i%%~xf"
        del %%f
    )
)

echo All benchmark runs complete.
endlocal
