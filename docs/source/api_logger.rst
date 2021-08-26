Logging
^^^^^^^^

Mechanica has a detailed logging system. Many internal methods will log
extensive details to either the clog (typically stderr) or a user
specified file path. The logging system can be configured to log events
at various levels of detail. All methods of the Logger are static,
they are available immediately upon loading the Mechanica package.

To display logging at the lowest level (``TRACE``), where every logging message is
displayed, is as simple as, ::

   import mechanica as mx
   mx.Logger.setLevel(mx.Logger.TRACE)

Enabling logging to terminal and disabling are also single commands, ::

   mx.Logger.enableConsoleLogging(mx.Logger.DEBUG)
   ...
   mx.Logger.disableConsoleLogging()

Messages can also be added to the log by logging level. ::

  mx.Logger.log(mx.Logger.FATAL, "A fatal message. This is the highest priority.")
  mx.Logger.log(mx.Logger.CRITICAL, "A critical message")
  mx.Logger.log(mx.Logger.ERROR, "An error message")
  mx.Logger.log(mx.Logger.WARNING, "A warning message")
  mx.Logger.log(mx.Logger.NOTICE, "A notice message")
  mx.Logger.log(mx.Logger.INFORMATION, "An informational message")
  mx.Logger.log(mx.Logger.DEBUG, "A debugging message.")
  mx.Logger.log(mx.Logger.TRACE,  "A tracing message. This is the lowest priority.")

.. autoclass:: Logger

.. autoclass:: MxLogger

   .. autoproperty:: CURRENT

   .. autoproperty:: FATAL

   .. autoproperty:: CRITICAL

   .. autoproperty:: ERROR

   .. autoproperty:: WARNING

   .. autoproperty:: NOTICE

   .. autoproperty:: INFORMATION

   .. autoproperty:: DEBUG

   .. autoproperty:: TRACE

   .. automethod:: setLevel

   .. automethod:: getLevel

   .. automethod:: disableLogging

   .. automethod:: enableConsoleLogging

   .. automethod:: disableConsoleLogging

   .. automethod:: enableFileLogging

   .. automethod:: disableFileLogging

   .. automethod:: getCurrentLevelAsString

   .. automethod:: getFileName

   .. automethod:: levelToString

   .. automethod:: stringToLevel

   .. automethod:: log
