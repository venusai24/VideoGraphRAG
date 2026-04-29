import "./globals.css";

export const metadata = {
  title: "VideoGraphRAG Portal",
  description: "Thin UI over the live VideoGraphRAG retrieval and answer pipeline."
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
