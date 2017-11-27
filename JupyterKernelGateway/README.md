# Testing Jupyter Kernel Gateway

Create conda environment

```
conda env create -f environment.yml
activate jupyter-nightly
```

Run lab

```
jupyter lab
```

Run gateway

```
jupyter kernelgateway --KernelGatewayApp.api=kernel_gateway.notebook_http --KernelGatewayApp.seed_uri=some/notebook/file.ipynb
```