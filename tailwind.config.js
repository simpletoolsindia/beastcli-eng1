/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,jsx}",
    "./components/**/*.{js,jsx}",
    "./lib/**/*.{js,jsx}"
  ],
  theme: {
    extend: {
      colors: {
        shell: {
          bg: "#0b141a",
          panel: "#111b21",
          soft: "#202c33",
          line: "#2a3942",
          text: "#e9edef",
          muted: "#8696a0",
          accent: "#25d366",
          bubbleSelf: "#005c4b",
          bubbleOther: "#202c33",
          bubbleTool: "#12212b",
          bubbleWarn: "#3b2c14"
        }
      },
      boxShadow: {
        phone: "0 32px 80px rgba(0, 0, 0, 0.35)"
      },
      backgroundImage: {
        dots: "radial-gradient(circle at 1px 1px, rgba(255,255,255,0.04) 1px, transparent 0)"
      }
    }
  },
  plugins: []
};
