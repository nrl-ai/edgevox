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

    // ``plan.md`` and ``reports/`` are internal planning artefacts we
    // don't want on the public site — excluded from the build.
    srcExclude: ["reports/**", "plan.md"],

    // Only the excluded planning artefacts should be tolerated as
    // dead-link targets; anything else going red must fail the build.
    ignoreDeadLinks: [/^\/plan/, /^\/reports\//],

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
        { text: "Documentation", link: "/documentation/" },
        {
          text: "Links",
          items: [
            { text: "GitHub", link: "https://github.com/vietanhdev/edgevox" },
            { text: "PyPI", link: "https://pypi.org/project/edgevox" },
          ],
        },
      ],

      sidebar: {
        "/documentation/": [
          {
            text: "Getting Started",
            items: [
              { text: "Introduction", link: "/documentation/" },
              { text: "Quick Start", link: "/documentation/quickstart" },
              { text: "Architecture", link: "/documentation/architecture" },
              { text: "Component Design", link: "/documentation/components" },
            ],
          },
          {
            text: "Features",
            items: [
              { text: "Languages", link: "/documentation/languages" },
              { text: "Voice Pipeline", link: "/documentation/pipeline" },
              { text: "Agents & Tools", link: "/documentation/agents" },
              { text: "TUI Commands", link: "/documentation/commands" },
              { text: "ROS2 Integration", link: "/documentation/ros2" },
              { text: "Chess Partner", link: "/documentation/chess" },
              { text: "RookApp (Desktop)", link: "/documentation/desktop" },
            ],
          },
          {
            text: "Harness Architecture",
            collapsed: false,
            items: [
              { text: "Agent loop", link: "/documentation/agent-loop" },
              { text: "Hooks", link: "/documentation/hooks" },
              { text: "Memory", link: "/documentation/memory" },
              { text: "Multi-agent", link: "/documentation/multiagent" },
              { text: "Interrupt & barge-in", link: "/documentation/interrupt" },
              { text: "Tool calling", link: "/documentation/tool-calling" },
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
