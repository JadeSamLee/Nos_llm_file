// import fs from "fs";
// // import axios from "axios";
// import { JWT } from "./constants";
const pinFileToIPFS = async (src) => {
  // const formData = new FormData();
  // const file = fs.createReadStream(src);
  // formData.append("file", file);
  // const metadata = JSON.stringify({
  //   name: "Certificate",
  // });
  // formData.append("pinataMetadata", metadata);
  // const options = JSON.stringify({
  //   cidVersion: 0,
  // });
  // formData.append("pinataOptions", options);
  // try {
  //   const res = await axios.post(
  //     "https://api.pinata.cloud/pinning/pinFileToIPFS",
  //     formData,
  //     {
  //       maxBodyLength: "Infinity",
  //       headers: {
  //         "Content-Type": `multipart/form-data; boundary=${formData._boundary}`,
  //         Authorization: JWT,
  //       },
  //     }
  //   );
  //   console.log(res.data);
  // } catch (error) {
  //   console.log(error);
  // }
};
export default pinFileToIPFS;
