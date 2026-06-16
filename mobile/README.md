# Plant Diagnosis Mobile

Expo mobile app for the bachelor thesis project. The phone sends the photo to the FastAPI API and displays the returned diagnosis.

## Quick Run With Expo Go

1. Install Node.js LTS from https://nodejs.org
2. Install Expo Go on the phone from the Play Store.
3. Run:

```powershell
cd C:\Users\Flavius\Desktop\Licenta\mobile
npm install
npx expo start
```

4. Scan the QR code with Expo Go.

If Windows asks about firewall access for Node.js/Expo, allow access on private networks.

If the phone is not on the same network as the laptop, use:

```powershell
npx expo start --tunnel
```

The server started by Codex uses port `8082`, so in Expo Go you can also enter manually:

```text
exp://192.168.100.38:8082
```

## API Configuration

By default, the app uses:

```text
http://16.16.99.225:8000
```

If you change the IP or move to an Elastic IP/domain, create the `.env` file:

```powershell
copy .env.example .env
```

and edit:

```text
EXPO_PUBLIC_API_BASE_URL=http://IP_SAU_DOMENIU:8000
```

## Installable APK Build

```powershell
cd C:\Users\Flavius\Desktop\Licenta\mobile
npm install
npx --yes eas-cli@latest login
npx --yes eas-cli@latest build:configure
npx --yes eas-cli@latest build -p android --profile preview
```

The `preview` profile generates an APK that can be installed directly on the phone.
