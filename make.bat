set DATASET=sdh_copper_80.arim

rem set FLAGS=-s --paper
rem set FLAGS=-s --noshow
set FLAGS=-s 

python -m arimtoolkit.adjust_filter %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.saturation %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.measure_probe_loc %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.plot_interfaces %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.measure_velocities_from_tfm %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.locate_defect %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.plot_interfaces %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.uniform_tfm %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.measure_tfm_intensity %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.adjust_toneburst %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.measure_attenuation_from_tfm %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.measure_model_scaling_from_tfm %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.tfm_walls %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.model_sensitivity %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.model_full %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.finalise_intensities %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.compare_tfm_intensities %DATASET% %FLAGS% || goto :error
python -m arimtoolkit.convert_svg %DATASET%/*.svg || goto :error
python -m arimtoolkit.sensitivity %DATASET% %FLAGS% || goto :error

echo Success
goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%
