# arimtoolkit

Scripts for analysing arim datasets in [ultrasonic testing](https://en.wikipedia.org/wiki/Ultrasonic_testing)

Dependencies: Python 3.6 or newer, [arim](https://github.com/ndtatbristol/arim), numpy, scipy, matplotlib, pyyaml, tqdm

[Configuration template ``conf.TEMPLATE.yaml``](https://github.com/nbud/arimtoolkit/blob/master/conf.TEMPLATE.yaml)

# Installation

    pip install git+https://github.com/nbud/arimtoolkit.git

Alternatively, [download zip of this repository](https://github.com/nbud/arimtoolkit/archive/master.zip), extract and run:

    python setup.py install

# Usage

Base structure of an arim analysis:

    MyDataset.arim/
        conf.yaml         # Adapted from conf.TEMPLATE.yaml
        conf.d/           # Placeholder directory for extra conf files

Results are stored in the ``.arim`` dataset, including its ``conf.d`` directory.

Most scripts have the following flags:

    -s, --save         Save results
    --noshow           Do not open matplotlib figures
    --paper            For paper-style figures (SVG/PDF, no title)


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
