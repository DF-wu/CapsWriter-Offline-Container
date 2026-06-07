const { createRunOncePlugin, withAndroidManifest } = require("expo/config-plugins");

const pkg = require("../package.json");

function withLocalHttpAndroid(config) {
  return withAndroidManifest(config, (nextConfig) => {
    const application = nextConfig.modResults.manifest.application?.[0];
    if (application) {
      application.$ = application.$ || {};
      application.$["android:usesCleartextTraffic"] = "true";
    }
    return nextConfig;
  });
}

module.exports = createRunOncePlugin(
  withLocalHttpAndroid,
  "with-local-http-android",
  pkg.version,
);
