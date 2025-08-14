import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd())

  return {
    define: {
      __API_BASE__: JSON.stringify(env.VITE_API_BASE)
    }
  }
})


