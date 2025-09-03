export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd())

  return {
    base: '/RugRadar/', // 👈 matches your repo name
    define: {
      __API_BASE__: JSON.stringify(env.VITE_API_BASE || 'https://woreffort.duckdns.org')
    },
    css: {
      postcss: {}
    }
  }
})



