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

@REM :: Loop to run Llama 2 inference on the CPU
@REM for /L %%i in (1,1,%num_runs%) do (
@REM     echo Running benchmark %%i...

@REM     :: Run the Python script 
@REM     python .\infer.py --model_name meta-llama/Llama-2-7b-hf --target cpu --model_dir llama2/quant --task benchmark --ort_trace --profile_file_prefix %prefixcpu%%%i

@REM     :: Find the file with the specified prefix
@REM     for %%f in ("%prefixcpu%%%i*") do (
@REM         echo Found file: %%f
@REM         :: Copy the file to the benchmark_runs directory with a unique name
@REM         copy "%%f" "benchmark_runs\%prefixcpu%%%i%%~xf"
@REM         del %%f
@REM     )
@REM )

:: Loop to run Llama 2 inference on the NPU
for /L %%i in (1,1,%num_runs%) do (
    echo Running benchmark %%i...

    :: Run the Python script 
    python .\infer.py --model_name meta-llama/Llama-2-7b-hf --target aie --model_dir llama2/quant --task benchmark --ort_trace --profile_file_prefix %prefixnpu%%%i

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
