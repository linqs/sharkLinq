const {FlatCompat} = require('@eslint/eslintrc');
const globals = require('globals');

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

module.exports = [
  {
    ignores: [
      '**/node_modules/**',
      '**/.git/**',
      'frontend/static/js/modules/shark_matcher/vendor/**',
    ],
  },
  ...compat.extends('google'),
  {
    files: ['frontend/eslint.config.js'],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: 'commonjs',
      globals: globals.node,
    },
  },
  {
    files: ['frontend/static/js/**/*.js', 'demo/*.js'],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: 'module',
      globals: globals.browser,
    },
  },
  {
    files: ['frontend/eslint.config.js', 'frontend/static/js/**/*.js', 'demo/*.js'],
    rules: {
      'max-len': ['error', {
        code: 100,
        ignoreUrls: true,
        ignoreStrings: true,
        ignoreRegExpLiterals: true,
        ignoreTemplateLiterals: true,
      }],
      'require-jsdoc': 'off',
      'valid-jsdoc': 'off',
      'no-undef': 'error',
    },
  },
  {
    files: ['frontend/static/js/modules/shark_matcher/**/*.js'],
    rules: {
      camelcase: 'off',
    },
  },
];
