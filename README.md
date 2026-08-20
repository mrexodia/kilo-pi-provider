# kilo-pi-provider

Kilo provider extension for Pi. Access 300+ AI models through the Kilo Gateway.

## Prerequisites

Install [Pi](https://pi.dev) (the coding agent CLI):

```bash
npm install -g @earendil-works/pi-coding-agent
```

## Installation

```bash
pi install git:github.com/mrexodia/kilo-pi-provider
```

## Usage

Start Pi as usual:

```bash
pi
```

Free models are available immediately. To access all models, log in with your [Kilo](https://kilo.ai) account:

```
/login
```

This opens your browser for device authorization. Once approved, all models become available in the model selector (`Ctrl+P`).

You can also set the `KILO_API_KEY` environment variable directly instead of using the login flow.
