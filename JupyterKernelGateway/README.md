# Testing Jupyter Kernel Gateway

Create conda environment

```
conda env create -f environment.yml
activate jupyter-kernel-gateway
```

Run lab

```
jupyter lab
```

Run gateway

```
jupyter kernelgateway --KernelGatewayApp.api=kernel_gateway.notebook_http --KernelGatewayApp.seed_uri=some/notebook/file.ipynb
```

Close conda env

```
deactivate
```

Delete conda env

```
conda env remove -n jupyter-kernel-gateway
```