# System Dynamics Model for River Resilience

## 1. Overview

This model is for evaluating several adaptation measures for the river basen flood risk management.

- **Tool**: This is made by Vensim PLE .
- **Main File**:  
  - `river_management.mdl`: main model file  
  - `river_management.2mdl`, `river_management.3mdl`: backup file(s)

## 2.Detailed Explanation

Please refer to the following link:
https://docs.google.com/document/d/116Xg9WkcorllC6vz6C-agFRKM3BDTerf/edit?usp=drive_link&ouid=107470865859100242765&rtpof=true&sd=true


## X. template-default

This is a template that is used by the `@sdeverywhere/create` package to generate a
new project that uses SDEverywhere.

The project includes:

- a build process that converts a Vensim model to a WebAssembly module that
  can run the model in any web browser or in a Node.js application
- a `config` directory that contains CSV files for configuring the generated
  model and application
- a "core" package that provides a clean JavaScript / TypeScript API around the
  WebAssembly model
- an "app" package containing a simple JavaScript / jQuery-based web application
  that can be used to exercise the model
- a local development mode (`npm run dev`) that allows for rapid prototyping
  of the model and app
- a "model-check" setup that allows for running checks and comparison tests using
  the generated model

## Quick Start

```sh
# Create a new project (you can also use yarn or pnpm here, if preferred).
# Be sure to choose the "Default" template.
npm create @sdeverywhere@latest

# Enter development mode for your model.  This will start a live
# development environment that will build a JavaScript version of the
# model and run checks on it any time you make changes to:
#   - the config files
#   - the Vensim model file (<name>.mdl)
#   - the model check definitions (model/checks/*.yaml)
#   - the model comparison definitions (model/comparisons/*.yaml)
npm run dev
```

## License

SDEverywhere is distributed under the MIT license. See `LICENSE` for more details.
