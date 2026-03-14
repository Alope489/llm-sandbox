FROM condaforge/miniforge3:26.1.0-0

# Install LAMMPS with Python interface from conda-forge in one layer,
# then clean all conda caches to minimise final image size.
RUN conda install -y -c conda-forge \
        "lammps=2024.08.29" \
        python=3.12 \
        numpy \
        scipy \
    && conda clean -afy \
    && find /opt/conda -name "*.pyc" -delete \
    && find /opt/conda -name "__pycache__" -type d -exec rm -rf {} +

# Application directory
WORKDIR /app

# Copy EAM potential files
COPY potentials/ /app/potentials/

# Copy the in-container calculation script
COPY elastic_tool.py /app/elastic_tool.py

# Copy AtomAgents input scripts (verbatim originals, patched at runtime)
COPY scripts/0_codes/ /app/scripts/

# Default entry point runs the elastic tool script
ENTRYPOINT ["python", "/app/elastic_tool.py"]