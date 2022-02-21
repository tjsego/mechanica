import mechanica as mx
print("Mechanica Version:")
print("version: ", mx.version.version)
print("build date: ", mx.version.build_date)
print("compiler: ", mx.version.compiler)
print("compiler_version: ", mx.version.compiler_version)
print("system_version: ", mx.version.system_version)

for k, v in mx.system.cpu_info().items():
    print("cpuinfo[", k, "]: ",  v)

for k, v in mx.system.compile_flags().items():
    print("compile_flags[", k, "]: ",  v)
