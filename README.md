# kilo-pi-provider

Kilo provider extension for Pi. Access Kilo Gateway models, including free models and authenticated models.

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

Free models are available immediately. To access the full model catalog, log in with your [Kilo](https://kilo.ai) account:

```
/login kilo
```

This opens your browser for device authorization. Once approved, the full catalog becomes available in Pi's model selector (`Ctrl+L`). Press `Ctrl+P` to cycle through configured models and `Shift+Tab` to cycle through the available thinking levels.

You can inspect the available models from the command line:

```bash
pi --provider kilo --list-models
```

When you sign in with `/login kilo`, the current account balance appears in Pi's footer while a Kilo model is active. It is hidden when you switch to another provider.

### API key authentication

You can use an API key instead of the browser login flow:

```bash
export KILO_API_KEY="your-key"
pi
```

### Configuration

The extension uses `https://api.kilo.ai` by default. Set `KILO_API_URL` to override the base URL for a compatible proxy or Kilo-compatible deployment:

```bash
export KILO_API_URL="https://api.kilo.ai"
```

### Updating

Update the installed extension with:

```bash
pi update git:github.com/mrexodia/kilo-pi-provider
```

Restart Pi after updating.

### Terms of service

Using Kilo means accepting the [Kilo Terms of Service](https://kilo.ai/terms).
