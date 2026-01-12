// @ts-check

/**
 * @returns {import('@docusaurus/types').Plugin}
 */
const nodePolyfillPlugin = () => ({
  name: 'node-polyfill-plugin',
  configureWebpack() {
    return {
      resolve: {
        fallback: {
          path: require.resolve('path-browserify'),
        },
      },
      plugins: [
        // This is a simple plugin to ensure require.resolveWeak exists
      ],
    };
  },
});

module.exports = nodePolyfillPlugin;