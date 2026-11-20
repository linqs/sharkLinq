// Original author Eriq Augustine (edulinq).

const LEVEL_OFF = 0;
const LEVEL_TRACE = 10;
const LEVEL_DEBUG = 20;
const LEVEL_INFO = 30;
const LEVEL_WARN = 40;
const LEVEL_ERROR = 50;
const LEVEL_FATAL = 60;

const LEVEL_TO_STRING = {
  [LEVEL_OFF]: 'off',
  [LEVEL_TRACE]: 'trace',
  [LEVEL_DEBUG]: 'debug',
  [LEVEL_INFO]: 'info',
  [LEVEL_WARN]: 'warn',
  [LEVEL_ERROR]: 'error',
  [LEVEL_FATAL]: 'fatal',
};

let records = [];

const SENSITIVE_LOG_KEY_PATTERN =
  /(authorization|credential|password|salt|secret|token|api_key|service_key)/i;
const REDACTED_LOG_VALUE = '[REDACTED]';

// Return a log-safe copy without mutating the caller's context object.
function sanitizeLogValue(value, seen = new WeakSet()) {
  if (value === null || typeof value !== 'object') {
    return value;
  }
  if (seen.has(value)) {
    return '[Circular]';
  }

  seen.add(value);
  const sanitized = Array.isArray(value) ? [] : {};
  for (const [key, nestedValue] of Object.entries(value)) {
    sanitized[key] = SENSITIVE_LOG_KEY_PATTERN.test(key) ?
      REDACTED_LOG_VALUE : sanitizeLogValue(nestedValue, seen);
  }
  seen.delete(value);
  return sanitized;
}

function init() {
  records = [];
}

function log(
    level, message, context = {}, notify = false, debugNotify = false) {
  if (level <= LEVEL_OFF) {
    return;
  }

  const record = {
    'time': Date,
    'level': LEVEL_TO_STRING[level],
    'raw-level': level,
    'message': message,
    'context': sanitizeLogValue(context),
    'notify': notify,
    'debug_notify': debugNotify,
  };

  if (level <= LEVEL_DEBUG) {
    console.debug(record);
  } else if (level <= LEVEL_INFO) {
    console.info(record);
  } else if (level <= LEVEL_WARN) {
    console.warn(record);
  } else {
    console.error(record);
  }

  if (message && message.stack && message.message) {
    // Explicitly log errors to get out all the info.
    console.log(message);
  }

  if (notify) {
    let msgStr = message;
    if (typeof message === 'object') {
      // If it's an APIError with a message, use that.
      if (message.message) msgStr = message.message;
    }

    if (msgStr === 'Unauthorized API call.') {
      alert(msgStr);
    } else {
      alert(`Error completing operation: ${msgStr}`);
    }
  }
  if (debugNotify) {
    alert(message);
  }
}

function trace(message, context = {}, notify = false, debugNotify = false) {
  log(LEVEL_TRACE, message, context, notify);
}

function debug(message, context = {}, notify = false, debugNotify = false) {
  log(LEVEL_DEBUG, message, context, notify);
}

function info(message, context = {}, notify = false, debugNotify = false) {
  log(LEVEL_INFO, message, context, notify);
}

function warn(message, context = {}, notify = true, debugNotify = false) {
  log(LEVEL_WARN, message, context, notify);
}

function error(message, context = {}, notify = true, debugNotify = true) {
  log(LEVEL_ERROR, message, context, notify);
}

function fatal(message, context = {}, notify = true, debugNotify = true) {
  log(LEVEL_FATAL, message, context, notify);
}

function getRecords() {
  return records.slice();
}

export {
  init,

  log,
  trace,
  debug,
  info,
  warn,
  error,
  fatal,

  getRecords,
};
