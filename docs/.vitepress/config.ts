import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";

export default withMermaid(
  defineConfig({
    title: "EdgeVox",
    description: "Sub-second local voice AI for robots and edge devices",
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
              { text: "Server API", link: "/reference/server-api" },
              { text: "Configuration", link: "/reference/config" },
              { text: "Language Config", link: "/reference/languages" },
            ],
          },
        ],
      },

      socialLinks: [
        { icon: "github", link: "https://github.com/vietanhdev/edgevox" },
      ],

      footer: {
        message: "Sub-second local voice AI for robots and edge devices",
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
