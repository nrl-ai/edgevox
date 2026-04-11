import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";

export default withMermaid(
  defineConfig({
    title: "EdgeVox",
    description: "Sub-second local voice AI for robots and edge devices",
    lang: "en-US",

    // Load docs from ./docs (synced from ../docs/)
    srcDir: ".",
    cleanUrls: true,

    appearance: "force-dark",

    head: [
      ["link", { rel: "icon", type: "image/svg+xml", href: "/logo.svg" }],
      [
        "meta",
        { name: "theme-color", content: "#00ff88" },
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
        {
          text: "Links",
          items: [
            { text: "GitHub", link: "https://github.com/edgevox-ai/edgevox" },
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
              { text: "TUI Commands", link: "/guide/commands" },
              { text: "ROS2 Integration", link: "/guide/ros2" },
            ],
          },
        ],
        "/reference/": [
          {
            text: "Reference",
            items: [
              { text: "CLI", link: "/reference/cli" },
              { text: "Configuration", link: "/reference/config" },
              { text: "Language Config", link: "/reference/languages" },
            ],
          },
        ],
      },

      socialLinks: [
        { icon: "github", link: "https://github.com/edgevox-ai/edgevox" },
      ],

      footer: {
        message: "Sub-second local voice AI",
        copyright: "MIT License",
      },

      editLink: {
        pattern: "https://github.com/edgevox-ai/edgevox/edit/main/docs/:path",
        text: "Edit this page on GitHub",
      },
    },
  }),
  {
    mermaid: {
      theme: "dark",
      themeVariables: {
        primaryColor: "#00ff88",
        primaryTextColor: "#c9d1d9",
        primaryBorderColor: "#1e3a2e",
        lineColor: "#00e5ff",
        secondaryColor: "#0d1117",
        tertiaryColor: "#111820",
        background: "#0a0e14",
        mainBkg: "#111820",
        nodeBorder: "#00ff88",
        clusterBkg: "#0d1117",
        clusterBorder: "#1e3a2e",
        titleColor: "#00ff88",
        edgeLabelBackground: "#0d1117",
        nodeTextColor: "#c9d1d9",
      },
    },
  }
);
