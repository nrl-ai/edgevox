import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";

export default withMermaid(
  defineConfig({
    title: "EdgeVox",
    description: "Offline voice agent framework for robots — agents, skills, workflows, 2D/3D simulation, sub-second voice pipeline",
    lang: "en-US",

    // Docs live directly in this directory
    srcDir: ".",
    cleanUrls: true,

    head: [
      ["link", { rel: "icon", type: "image/svg+xml", href: "/logo.svg" }],
      [
        "meta",
        { name: "theme-color", content: "#c96442" },
      ],
    ],

    themeConfig: {
      logo: "/logo.svg",
      siteTitle: "EdgeVox",

      search: {
        provider: "local",
      },

      nav: [
        { text: "Guide", link: "/guide/" },
        { text: "Reference", link: "/reference/cli" },
        { text: "ADRs", link: "/adr/001-cancel-token-plumbing" },
        {
          text: "Links",
          items: [
            { text: "GitHub", link: "https://github.com/vietanhdev/edgevox" },
            { text: "PyPI", link: "https://pypi.org/project/edgevox" },
          ],
        },
      ],

      sidebar: {
        "/guide/": [
          {
            text: "Getting Started",
            items: [
              { text: "Introduction", link: "/guide/" },
              { text: "Quick Start", link: "/guide/quickstart" },
              { text: "Architecture", link: "/guide/architecture" },
            ],
          },
          {
            text: "Features",
            items: [
              { text: "Languages", link: "/guide/languages" },
              { text: "Voice Pipeline", link: "/guide/pipeline" },
              { text: "Agents & Tools", link: "/guide/agents" },
              { text: "TUI Commands", link: "/guide/commands" },
              { text: "ROS2 Integration", link: "/guide/ros2" },
              { text: "Chess Partner", link: "/guide/chess" },
              { text: "RookApp (Desktop)", link: "/guide/desktop" },
            ],
          },
          {
            text: "Harness Architecture",
            collapsed: false,
            items: [
              { text: "Agent loop", link: "/guide/agent-loop" },
              { text: "Hooks", link: "/guide/hooks" },
              { text: "Memory", link: "/guide/memory" },
              { text: "Multi-agent", link: "/guide/multiagent" },
              { text: "Interrupt & barge-in", link: "/guide/interrupt" },
              { text: "Tool calling", link: "/guide/tool-calling" },
            ],
          },
        ],
        "/reference/": [
          {
            text: "Reference",
            items: [
              { text: "CLI", link: "/reference/cli" },
              { text: "Server API", link: "/reference/server-api" },
              { text: "Configuration", link: "/reference/config" },
              { text: "Language Config", link: "/reference/languages" },
            ],
          },
        ],
        "/adr/": [
          {
            text: "Architecture Decisions",
            items: [
              { text: "001 — Cancel-token plumbing", link: "/adr/001-cancel-token-plumbing" },
              { text: "002 — Typed ctx + hook-owned state", link: "/adr/002-typed-ctx-hook-state" },
              { text: "003 — GBNF tool decoding", link: "/adr/003-grammar-constrained-decoding" },
            ],
          },
        ],
      },

      socialLinks: [
        { icon: "github", link: "https://github.com/vietanhdev/edgevox" },
      ],

      footer: {
        message: "Offline voice agent framework for robots",
        copyright: "MIT License",
      },

      editLink: {
        pattern: "https://github.com/vietanhdev/edgevox/edit/main/docs/:path",
        text: "Edit this page on GitHub",
      },
    },
  }),
  {
    mermaid: {
      theme: "neutral",
      themeVariables: {
        primaryColor: "#f5ebe0",
        primaryTextColor: "#1a1613",
        primaryBorderColor: "#c96442",
        lineColor: "#c96442",
        secondaryColor: "#faf7f2",
        tertiaryColor: "#f5f0e8",
        background: "#faf7f2",
        mainBkg: "#f5ebe0",
        nodeBorder: "#c96442",
        clusterBkg: "#faf7f2",
        clusterBorder: "#d4c9b9",
        titleColor: "#1a1613",
        edgeLabelBackground: "#faf7f2",
        nodeTextColor: "#1a1613",
      },
    },
  }
);
