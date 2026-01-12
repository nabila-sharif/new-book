import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/blog',
    component: ComponentCreator('/blog', 'df4'),
    exact: true
  },
  {
    path: '/blog/archive',
    component: ComponentCreator('/blog/archive', '182'),
    exact: true
  },
  {
    path: '/blog/authors',
    component: ComponentCreator('/blog/authors', '0b7'),
    exact: true
  },
  {
    path: '/blog/tags',
    component: ComponentCreator('/blog/tags', '287'),
    exact: true
  },
  {
    path: '/blog/tags/welcome',
    component: ComponentCreator('/blog/tags/welcome', '358'),
    exact: true
  },
  {
    path: '/blog/welcome',
    component: ComponentCreator('/blog/welcome', 'bbd'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '9b5'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '04c'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '39e'),
            routes: [
              {
                path: '/docs/',
                component: ComponentCreator('/docs/', '0ee'),
                exact: true
              },
              {
                path: '/docs/docs/intro',
                component: ComponentCreator('/docs/docs/intro', '2e5'),
                exact: true
              },
              {
                path: '/docs/module-2',
                component: ComponentCreator('/docs/module-2', '162'),
                exact: true,
                sidebar: "module2Sidebar"
              },
              {
                path: '/docs/module-2/chapter-1-content',
                component: ComponentCreator('/docs/module-2/chapter-1-content', 'e41'),
                exact: true,
                sidebar: "module2Sidebar"
              },
              {
                path: '/docs/module-2/chapter-2-content',
                component: ComponentCreator('/docs/module-2/chapter-2-content', '2d8'),
                exact: true,
                sidebar: "module2Sidebar"
              },
              {
                path: '/docs/module-2/chapter-3-content',
                component: ComponentCreator('/docs/module-2/chapter-3-content', '3da'),
                exact: true,
                sidebar: "module2Sidebar"
              },
              {
                path: '/docs/module-3',
                component: ComponentCreator('/docs/module-3', 'f10'),
                exact: true,
                sidebar: "module3Sidebar"
              },
              {
                path: '/docs/module-3/chapter-1-content',
                component: ComponentCreator('/docs/module-3/chapter-1-content', '9e2'),
                exact: true,
                sidebar: "module3Sidebar"
              },
              {
                path: '/docs/module-3/chapter-2-content',
                component: ComponentCreator('/docs/module-3/chapter-2-content', '4f4'),
                exact: true,
                sidebar: "module3Sidebar"
              },
              {
                path: '/docs/module-3/chapter-3-content',
                component: ComponentCreator('/docs/module-3/chapter-3-content', 'c5d'),
                exact: true,
                sidebar: "module3Sidebar"
              },
              {
                path: '/docs/module-4/',
                component: ComponentCreator('/docs/module-4/', '075'),
                exact: true,
                sidebar: "module4Sidebar"
              },
              {
                path: '/docs/module-4/chapter-1-content',
                component: ComponentCreator('/docs/module-4/chapter-1-content', '73a'),
                exact: true,
                sidebar: "module4Sidebar"
              },
              {
                path: '/docs/module-4/chapter-2-content',
                component: ComponentCreator('/docs/module-4/chapter-2-content', '157'),
                exact: true,
                sidebar: "module4Sidebar"
              },
              {
                path: '/docs/module-4/chapter-3-content',
                component: ComponentCreator('/docs/module-4/chapter-3-content', '5c5'),
                exact: true,
                sidebar: "module4Sidebar"
              },
              {
                path: '/docs/module-4/style-guide',
                component: ComponentCreator('/docs/module-4/style-guide', '5b5'),
                exact: true,
                sidebar: "module4Sidebar"
              },
              {
                path: '/docs/physical-ai-humanoid-robotics/',
                component: ComponentCreator('/docs/physical-ai-humanoid-robotics/', 'da2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/physical-ai-humanoid-robotics/chapter1-ros2-fundamentals',
                component: ComponentCreator('/docs/physical-ai-humanoid-robotics/chapter1-ros2-fundamentals', '9fa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/physical-ai-humanoid-robotics/chapter1-ros2-fundamentals-test',
                component: ComponentCreator('/docs/physical-ai-humanoid-robotics/chapter1-ros2-fundamentals-test', '746'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/physical-ai-humanoid-robotics/chapter2-ros2-python-communication',
                component: ComponentCreator('/docs/physical-ai-humanoid-robotics/chapter2-ros2-python-communication', '59b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/physical-ai-humanoid-robotics/chapter2-ros2-python-communication-test',
                component: ComponentCreator('/docs/physical-ai-humanoid-robotics/chapter2-ros2-python-communication-test', '0b3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/physical-ai-humanoid-robotics/chapter3-urdf-humanoid-structure',
                component: ComponentCreator('/docs/physical-ai-humanoid-robotics/chapter3-urdf-humanoid-structure', '28a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/physical-ai-humanoid-robotics/chapter3-urdf-humanoid-structure-test',
                component: ComponentCreator('/docs/physical-ai-humanoid-robotics/chapter3-urdf-humanoid-structure-test', '7b9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/physical-ai-humanoid-robotics/module1-intro',
                component: ComponentCreator('/docs/physical-ai-humanoid-robotics/module1-intro', 'fa0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/physical-ai-humanoid-robotics/quickstart',
                component: ComponentCreator('/docs/physical-ai-humanoid-robotics/quickstart', 'fbe'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
