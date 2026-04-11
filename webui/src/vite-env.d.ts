/// <reference types="vite/client" />

declare module "*.worklet.js?url" {
  const url: string;
  export default url;
}
