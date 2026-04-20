import "./globals.css";

export const metadata = {
  title: "Dataset Reviewer",
  description: "WhatsApp-style JSONL dataset reviewer for tool-calling samples."
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
