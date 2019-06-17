# arimtoolkit

Scripts for analysing arim datasets

Dependencies: [arim](https://github.com/ndtatbristol/arim), numpy, scipy, matplotlib, pyyaml, tqdm

[Configuration template](https://github.com/nbud/arimtoolkit/blob/master/conf.TEMPLATE.yaml)

# Usage

## TFM on unknown dataset

    python -m arimtoolkit.measure_probe_loc MyDataset.arim --save
    python -m arimtoolkit.plot_interfaces MyDataset.arim --save
    python -m arimtoolkit.measure_velocities_from_tfm MyDataset.arim --save
    python -m arimtoolkit.uniform_tfm MyDataset.arim --save

## Compare experimental and modelled defect intensities

    python -m arimtoolkit.saturation MyDataset.arim --save
    python -m arimtoolkit.measure_probe_loc MyDataset.arim --save
    python -m arimtoolkit.plot_interfaces MyDataset.arim --save
    python -m arimtoolkit.measure_velocities_from_tfm MyDataset.arim --save
    python -m arimtoolkit.locate_defect MyDataset.arim --save
    python -m arimtoolkit.plot_interfaces MyDataset.arim --save
    python -m arimtoolkit.uniform_tfm MyDataset.arim --save
    python -m arimtoolkit.measure_tfm_intensity MyDataset.arim --save
    python -m arimtoolkit.adjust_toneburst MyDataset.arim --save
    python -m arimtoolkit.measure_attenuation_from_tfm MyDataset.arim --save
    python -m arimtoolkit.measure_model_scaling_from_tfm MyDataset.arim --save
    python -m arimtoolkit.tfm_walls MyDataset.arim --save
    python -m arimtoolkit.model_sensitivity MyDataset.arim --save
    python -m arimtoolkit.model_full MyDataset.arim --save
    python -m arimtoolkit.collect_intensities MyDataset.arim --save
    python -m arimtoolkit.compare_tfm_intensities MyDataset.arim --save

## Run sensitivity

    python -m arimtoolkit.sensitivity MyDataset.arim --save
