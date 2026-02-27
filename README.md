# kilo-pi-provider

[Kilo](https://kilo.ai) provider extension for [Pi](https://pi.dev). Access 500+ AI models through the Kilo Gateway.

## Prerequisites

Install Pi (the coding agent CLI):

```bash
npm install -g @mariozechner/pi-coding-agent
```

Create a free Kilo account at [kilo.ai](https://kilo.ai) (optional -- free models work without an account).

## Installation

```bash
pi install git:github.com/mrexodia/kilo-pi-provider
```

## Usage

Start Pi as usual:

```bash
pi
```

Free models are available immediately. To access all 500+ models, log in with your Kilo account:

```
/login kilo
```

This opens your browser for device authorization. Once approved, all models become available in the model selector (`ctrl+l`).

You can also set the `KILO_API_KEY` environment variable directly instead of using the login flow.

## License

MIT
