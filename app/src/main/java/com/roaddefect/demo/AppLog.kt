package com.roaddefect.demo

import android.util.Log

/**
 * Wraps android.util.Log so debug logging can be globally enabled/disabled
 * from one place (AppConfig.DEBUG_LOGGING_ENABLED) instead of commenting out
 * individual Log.d calls scattered across files.
 *
 * Usage: replace Log.d(tag, msg) with AppLog.d(tag, msg) throughout the codebase.
 * Errors/warnings always log regardless of the flag, since those matter even
 * in release builds.
 */
object AppLog {

    fun d(tag: String, message: String) {
        if (AppConfig.DEBUG_LOGGING_ENABLED) {
            Log.d(tag, message)
        }
    }

    fun w(tag: String, message: String) {
        Log.w(tag, message)
    }

    fun e(tag: String, message: String) {
        Log.e(tag, message)
    }
}