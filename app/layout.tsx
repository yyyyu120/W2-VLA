import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const ogImage = `${protocol}://${host}/assets/figures/teaser.png`;

  return {
    title: "W²-VLA · World-to-Wrist",
    description: "Task-conditioned future wrist modeling for fine-grained robot manipulation.",
    openGraph: {
      title: "W²-VLA · World-to-Wrist",
      description: "Task-conditioned future wrist modeling for fine-grained robot manipulation.",
      type: "website",
      images: [{ url: ogImage, width: 2400, height: 1028, alt: "W²-VLA World-to-Wrist method teaser" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "W²-VLA · World-to-Wrist",
      description: "Task-conditioned future wrist modeling for fine-grained robot manipulation.",
      images: [ogImage],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
